package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"github.com/gorilla/websocket"
	"github.com/rs/cors"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

const (
	modelPath     = "yolo26n.onnx"
	inputSize     = 640
	planeSize     = inputSize * inputSize // 640×640
	confThreshold = 0.4
	listenAddr    = ":8080"
)

// ── 타입 ────────────────────────────────────────────────────────────────────

type Detection struct {
	Box   [4]int  `json:"box"`
	Score float64 `json:"score"`
	Label int     `json:"label"`
	Name  string  `json:"name"`
}

type wsResponse struct {
	Detections []Detection `json:"detections"`
}
type wsError struct {
	Error string `json:"error"`
}

// ── Server ───────────────────────────────────────────────────────────────────
// Server holds all shared inference state and pools.
// Methods are the HTTP/WS handlers, so the mux wires directly to methods.

type Server struct {
	session    *ort.DynamicAdvancedSession
	classNames map[int]string
	upgrader   websocket.Upgrader
	inputPool  sync.Pool // *[]float32 len=3*planeSize — reused across frames
	bufPool    sync.Pool // *bytes.Buffer — reused per connection for JSON
}

func newServer(session *ort.DynamicAdvancedSession, classNames map[int]string) *Server {
	s := &Server{
		session:    session,
		classNames: classNames,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1 << 20,
			WriteBufferSize: 1 << 20,
			CheckOrigin:     func(*http.Request) bool { return true },
		},
	}
	s.inputPool.New = func() any {
		buf := make([]float32, 3*planeSize)
		return &buf
	}
	s.bufPool.New = func() any { return new(bytes.Buffer) }
	return s
}

func (s *Server) className(label int) string {
	if name, ok := s.classNames[label]; ok {
		return name
	}
	return fmt.Sprintf("cls%d", label)
}

// ── 후처리 ──────────────────────────────────────────────────────────────────
// YOLO26 출력 형태: (1, N, 6) 또는 (N, 6) — [x1, y1, x2, y2, score, label]

func (s *Server) postprocess(data []float32, shape ort.Shape, scaleX, scaleY float32) []Detection {
	var n int64
	switch len(shape) {
	case 3:
		n = shape[1]
	case 2:
		n = shape[0]
	default:
		return nil
	}

	out := make([]Detection, 0, n) // capacity hint avoids repeated reallocation
	for i := int64(0); i < n; i++ {
		row := data[i*6 : i*6+6]
		score := row[4]
		if score < confThreshold {
			continue
		}
		label := int(row[5])
		out = append(out, Detection{
			Box:   [4]int{int(row[0] * scaleX), int(row[1] * scaleY), int(row[2] * scaleX), int(row[3] * scaleY)},
			Score: float64(int64(score*10000+0.5)) / 10000, // round to 4 dp, no math import
			Label: label,
			Name:  s.className(label),
		})
	}
	return out
}

// ── 추론 ─────────────────────────────────────────────────────────────────────

func (s *Server) infer(frameBytes []byte) ([]Detection, error) {
	img, err := gocv.IMDecode(frameBytes, gocv.IMReadColor)
	if err != nil || img.Empty() {
		return nil, fmt.Errorf("image decode failed")
	}
	defer img.Close()

	scaleX := float32(img.Cols()) / inputSize
	scaleY := float32(img.Rows()) / inputSize

	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(img, &resized, image.Point{X: inputSize, Y: inputSize}, 0, 0, gocv.InterpolationLinear)

	// HWC (BGR interleaved) → CHW float32/255 using a pooled buffer.
	// Single flat loop instead of triple-nested: sequential reads from raw,
	// predictable writes into three contiguous planes of inp.
	inpPtr := s.inputPool.Get().(*[]float32)
	inp := *inpPtr
	raw := resized.ToBytes()
	for i := 0; i < planeSize; i++ {
		off := i * 3
		inp[i] = float32(raw[off]) / 255.0
		inp[planeSize+i] = float32(raw[off+1]) / 255.0
		inp[2*planeSize+i] = float32(raw[off+2]) / 255.0
	}

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, inputSize, inputSize), inp)
	if err != nil {
		s.inputPool.Put(inpPtr)
		return nil, fmt.Errorf("tensor creation: %w", err)
	}

	outputs := make([]ort.Value, 1)
	err = s.session.Run([]ort.Value{inputTensor}, outputs)
	inputTensor.Destroy()
	s.inputPool.Put(inpPtr) // safe: tensor destroyed, buffer no longer referenced
	if err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			o.Destroy()
		}
	}()

	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected output tensor type")
	}
	return s.postprocess(outTensor.GetData(), outTensor.GetShape(), scaleX, scaleY), nil
}

// ── 핸들러 ───────────────────────────────────────────────────────────────────

func (s *Server) healthCheck(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":       "ok",
		"model_loaded": s.session != nil,
	})
}

func (s *Server) wsStream(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		slog.Error("ws upgrade", "err", err)
		return
	}
	defer conn.Close()

	buf := s.bufPool.Get().(*bytes.Buffer)
	defer s.bufPool.Put(buf)

	for {
		msgType, data, err := conn.ReadMessage()
		if err != nil {
			break
		}
		if msgType != websocket.BinaryMessage {
			continue
		}

		buf.Reset()
		detections, err := s.infer(data)
		if err != nil {
			_ = json.NewEncoder(buf).Encode(wsError{err.Error()})
		} else {
			_ = json.NewEncoder(buf).Encode(wsResponse{detections})
		}
		if err := conn.WriteMessage(websocket.TextMessage, buf.Bytes()); err != nil {
			break
		}
	}
}

// ── ONNX 메타데이터 파서 ──────────────────────────────────────────────────────
// ultralytics ONNX export는 ModelProto.metadata_props (field 14)에
// 클래스 이름을 저장한다. 외부 proto 라이브러리 없이 최소 파서로 읽는다.

func readVarint(data []byte, pos int) (uint64, int) {
	var result uint64
	var shift uint
	for pos < len(data) {
		b := data[pos]
		pos++
		result |= uint64(b&0x7F) << shift
		if b&0x80 == 0 {
			return result, pos
		}
		shift += 7
	}
	return result, pos
}

func parseStringStringEntry(data []byte) (key, val string) {
	pos := 0
	for pos < len(data) {
		tag, newPos := readVarint(data, pos)
		if newPos <= pos {
			break
		}
		pos = newPos
		fieldNum := tag >> 3
		wireType := tag & 0x7
		if wireType != 2 {
			break
		}
		length, newPos := readVarint(data, pos)
		pos = newPos
		end := pos + int(length)
		if end > len(data) {
			break
		}
		s := string(data[pos:end])
		pos = end
		switch fieldNum {
		case 1:
			key = s
		case 2:
			val = s
		}
	}
	return
}

func parseONNXMetadata(path string) map[string]string {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	result := make(map[string]string)
	pos := 0
	for pos < len(data) {
		tag, newPos := readVarint(data, pos)
		if newPos <= pos {
			break
		}
		pos = newPos
		fieldNum := tag >> 3
		wireType := tag & 0x7

		switch wireType {
		case 0:
			_, pos = readVarint(data, pos)
		case 1:
			pos += 8
			if pos > len(data) {
				return result
			}
		case 2:
			length, newPos := readVarint(data, pos)
			pos = newPos
			end := pos + int(length)
			if end > len(data) {
				return result
			}
			if fieldNum == 14 { // metadata_props
				k, v := parseStringStringEntry(data[pos:end])
				if k != "" {
					result[k] = v
				}
			}
			pos = end
		case 5:
			pos += 4
			if pos > len(data) {
				return result
			}
		default:
			return result
		}
	}
	return result
}

// parseClassNames parses the ultralytics Python-dict string:
// "{0: 'person', 1: 'bicycle', ...}" → map[int]string
func parseClassNames(raw string) map[int]string {
	result := make(map[int]string)
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "{")
	raw = strings.TrimSuffix(raw, "}")

	var entries []string
	var cur strings.Builder
	inQuote := false
	var quoteChar byte
	for i := 0; i < len(raw); i++ {
		ch := raw[i]
		switch {
		case !inQuote && (ch == '\'' || ch == '"'):
			inQuote = true
			quoteChar = ch
			cur.WriteByte(ch)
		case inQuote && ch == quoteChar:
			inQuote = false
			cur.WriteByte(ch)
		case !inQuote && ch == ',':
			entries = append(entries, cur.String())
			cur.Reset()
		default:
			cur.WriteByte(ch)
		}
	}
	if cur.Len() > 0 {
		entries = append(entries, cur.String())
	}

	for _, entry := range entries {
		entry = strings.TrimSpace(entry)
		colonIdx := strings.Index(entry, ":")
		if colonIdx < 0 {
			continue
		}
		keyStr := strings.TrimSpace(entry[:colonIdx])
		valStr := strings.Trim(strings.TrimSpace(entry[colonIdx+1:]), "'\"")
		idx, err := strconv.Atoi(keyStr)
		if err != nil {
			continue
		}
		result[idx] = valStr
	}
	return result
}

// ── 메인 ────────────────────────────────────────────────────────────────────

func main() {
	// 라이브러리 파일 이름을 명시적으로 지정 (Docker 기준)
	ort.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so")

	if err := ort.InitializeEnvironment(); err != nil {
		slog.Error("ORT init failed", "err", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	inputInfo, outputInfo, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		slog.Error("model info query failed", "err", err)
		os.Exit(1)
	}
	inputNames := make([]string, len(inputInfo))
	for i, info := range inputInfo {
		inputNames[i] = info.Name
	}
	outputNames := make([]string, len(outputInfo))
	for i, info := range outputInfo {
		outputNames[i] = info.Name
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		slog.Error("session create failed", "err", err)
		os.Exit(1)
	}
	defer session.Destroy()

	classNames := make(map[int]string)
	if meta := parseONNXMetadata(modelPath); meta != nil {
		if namesStr, ok := meta["names"]; ok {
			classNames = parseClassNames(namesStr)
		}
	}
	slog.Info("model loaded", "path", modelPath, "classes", len(classNames))

	srv := newServer(session, classNames)

	mux := http.NewServeMux()
	mux.HandleFunc("/", srv.healthCheck)
	mux.HandleFunc("/ws/stream", srv.wsStream)

	// Cloud Run injects $PORT (typically 8080); fall back to the default.
	port := os.Getenv("PORT")
	if port == "" {
		port = strings.TrimPrefix(listenAddr, ":")
	}
	addr := ":" + port

	httpSrv := &http.Server{
		Addr:    addr,
		Handler: cors.AllowAll().Handler(mux),
	}

	// Graceful shutdown on Ctrl-C / SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		_ = httpSrv.Shutdown(context.Background())
	}()

	slog.Info("server started", "addr", addr)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}
