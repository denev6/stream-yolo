import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:web_socket_channel/web_socket_channel.dart';

Uint8List _encodeFrame(Map<String, dynamic> p) {
  final int width = p['width'] as int;
  final int height = p['height'] as int;
  final int rotation = p['rotation'] as int;
  final String format = p['format'] as String;

  img.Image frame;

  if (format == 'yuv420') {
    final Uint8List yBytes = p['y'] as Uint8List;
    final Uint8List uBytes = p['u'] as Uint8List;
    final Uint8List vBytes = p['v'] as Uint8List;
    final int yStride = p['yStride'] as int;
    final int uvStride = p['uvStride'] as int;
    final int uvPixelStride = p['uvPixelStride'] as int;

    frame = img.Image(width: width, height: height, numChannels: 3);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final yVal = yBytes[y * yStride + x];
        final uvIdx = (y ~/ 2) * uvStride + (x ~/ 2) * uvPixelStride;
        final uVal = uBytes[uvIdx];
        final vVal = vBytes[uvIdx];

        final c = yVal - 16;
        final d = uVal - 128;
        final e = vVal - 128;
        final r = ((298 * c + 409 * e + 128) >> 8).clamp(0, 255);
        final g = ((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255);
        final b = ((298 * c + 516 * d + 128) >> 8).clamp(0, 255);

        frame.setPixelRgb(x, y, r, g, b);
      }
    }
  } else if (format == 'bgra') {
    final Uint8List bytes = p['bytes'] as Uint8List;
    final int rowStride = p['rowStride'] as int;

    if (rowStride == width * 4) {
      frame = img.Image.fromBytes(
        width: width,
        height: height,
        bytes: bytes.buffer,
        numChannels: 4,
        order: img.ChannelOrder.bgra,
      );
    } else {
      frame = img.Image(width: width, height: height, numChannels: 3);
      for (int y = 0; y < height; y++) {
        final int rowOffset = y * rowStride;
        for (int x = 0; x < width; x++) {
          final int offset = rowOffset + x * 4;
          frame.setPixelRgb(
            x,
            y,
            bytes[offset + 2],
            bytes[offset + 1],
            bytes[offset],
          );
        }
      }
    }
  } else {
    return p['bytes'] as Uint8List;
  }

  img.Image out;
  if (width > height && rotation != 0) {
    out = img.copyRotate(frame, angle: rotation);
  } else {
    out = frame;
  }

  return Uint8List.fromList(img.encodeJpg(out, quality: 50));
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Stream',
      theme: ThemeData.dark(useMaterial3: true),
      home: ConnectionScreen(cameras: cameras),
    );
  }
}

class ConnectionScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const ConnectionScreen({super.key, required this.cameras});

  @override
  State<ConnectionScreen> createState() => _ConnectionScreenState();
}

class _ConnectionScreenState extends State<ConnectionScreen> {
  final _ipCtrl = TextEditingController(text: '192.168.0.1');
  final _portCtrl = TextEditingController(text: '8001');

  void _connect() {
    final ip = _ipCtrl.text.trim();
    final port = _portCtrl.text.trim();
    if (ip.isEmpty || port.isEmpty) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) =>
            StreamScreen(cameras: widget.cameras, ip: ip, port: port),
      ),
    );
  }

  @override
  void dispose() {
    _ipCtrl.dispose();
    _portCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('YOLO Stream')),
      body: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.videocam, size: 64),
            const SizedBox(height: 32),
            TextField(
              controller: _ipCtrl,
              decoration: const InputDecoration(
                labelText: 'Server IP',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.computer),
              ),
              keyboardType: TextInputType.url,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _portCtrl,
              decoration: const InputDecoration(
                labelText: 'Port',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.settings_ethernet),
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: _connect,
                icon: const Icon(Icons.play_arrow),
                label: const Text('Connect & Stream'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class Detection {
  final List<double> box;
  final double score;
  final String name;

  Detection({required this.box, required this.score, required this.name});

  factory Detection.fromJson(Map<String, dynamic> j) => Detection(
    box: (j['box'] as List).map((e) => (e as num).toDouble()).toList(),
    score: (j['score'] as num).toDouble(),
    name: j['name'] as String,
  );
}

class StreamScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  final String ip;
  final String port;

  const StreamScreen({
    super.key,
    required this.cameras,
    required this.ip,
    required this.port,
  });

  @override
  State<StreamScreen> createState() => _StreamScreenState();
}

class _StreamScreenState extends State<StreamScreen> {
  CameraController? _cam;
  WebSocketChannel? _ws;
  List<Detection> _detections = [];
  bool _frameInFlight = false;
  String? _error;
  int _fps = 0;
  int _frameCount = 0;
  late Timer _fpsTimer;
  Size? _serverImageSize;

  @override
  void initState() {
    super.initState();
    _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {
          _fps = _frameCount;
          _frameCount = 0;
        });
      }
    });
    _initCamera();
  }

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() => _error = 'No camera available');
      return;
    }
    _cam = CameraController(
      widget.cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    try {
      await _cam!.initialize();
      if (mounted) {
        setState(() {});
        _connectWs();
      }
    } catch (e) {
      setState(() => _error = 'Camera init failed: $e');
    }
  }

  void _connectWs() {
    final uri = Uri.parse('ws://${widget.ip}:${widget.port}/ws/stream');
    try {
      _ws = WebSocketChannel.connect(uri);
      _ws!.stream.listen(
        (msg) {
          _frameInFlight = false;
          if (msg is String) {
            final decoded = jsonDecode(msg) as Map<String, dynamic>;
            final list = decoded['detections'] as List<dynamic>;
            if (mounted) {
              setState(() {
                _detections = list
                    .map((d) => Detection.fromJson(d as Map<String, dynamic>))
                    .toList();
              });
            }
          }
        },
        onError: (e) {
          if (mounted) setState(() => _error = 'WebSocket error: $e');
          _stopStream();
        },
        onDone: () {
          if (mounted) setState(() => _error = 'Server disconnected');
          _stopStream();
        },
      );
      _startStream();
    } catch (e) {
      setState(() => _error = 'Connection failed: $e');
    }
  }

  void _startStream() {
    _cam!.startImageStream((CameraImage image) async {
      if (_serverImageSize == null) {
        if (mounted) {
          setState(() {
            if (image.width > image.height &&
                widget.cameras.first.sensorOrientation != 0) {
              _serverImageSize = Size(
                image.height.toDouble(),
                image.width.toDouble(),
              );
            } else {
              _serverImageSize = Size(
                image.width.toDouble(),
                image.height.toDouble(),
              );
            }
          });
        }
      }

      if (_frameInFlight || _ws == null) return;
      _frameInFlight = true;
      try {
        final jpeg = await _buildParams(image);
        _ws!.sink.add(jpeg);
        _frameCount++;
      } catch (_) {
        _frameInFlight = false;
      }
    });
  }

  void _stopStream() {
    _cam?.stopImageStream();
  }

  Future<Uint8List> _buildParams(CameraImage image) {
    final rotation = widget.cameras.first.sensorOrientation;
    final Map<String, dynamic> params;

    final fmt = image.format.group;
    if (fmt == ImageFormatGroup.yuv420) {
      params = {
        'format': 'yuv420',
        'width': image.width,
        'height': image.height,
        'rotation': rotation,
        'y': image.planes[0].bytes,
        'u': image.planes[1].bytes,
        'v': image.planes[2].bytes,
        'yStride': image.planes[0].bytesPerRow,
        'uvStride': image.planes[1].bytesPerRow,
        'uvPixelStride': image.planes[1].bytesPerPixel ?? 1,
      };
    } else if (fmt == ImageFormatGroup.bgra8888) {
      params = {
        'format': 'bgra',
        'width': image.width,
        'height': image.height,
        'rotation': rotation,
        'bytes': image.planes[0].bytes,
        'rowStride': image.planes[0].bytesPerRow,
      };
    } else {
      params = {
        'format': 'jpeg',
        'bytes': image.planes[0].bytes,
        'width': image.width,
        'height': image.height,
        'rotation': 0,
      };
    }

    return compute(_encodeFrame, params);
  }

  @override
  void dispose() {
    _fpsTimer.cancel();
    _ws?.sink.close();
    _cam?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: Text('${widget.ip}:${widget.port}')),
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text(_error!, style: const TextStyle(color: Colors.red)),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Back'),
              ),
            ],
          ),
        ),
      );
    }

    if (_cam == null ||
        !_cam!.value.isInitialized ||
        _serverImageSize == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      backgroundColor: Colors.black, // 여백을 검정색으로 처리
      appBar: AppBar(
        title: Text('${widget.ip}:${widget.port}'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(child: Text('$_fps fps')),
          ),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          Center(
            child: AspectRatio(
              aspectRatio: _serverImageSize!.width / _serverImageSize!.height,
              child: CameraPreview(_cam!),
            ),
          ),
          LayoutBuilder(
            builder: (ctx, constraints) {
              return CustomPaint(
                painter: DetectionPainter(
                  detections: _detections,
                  serverImageSize: _serverImageSize!,
                  displaySize: Size(
                    constraints.maxWidth,
                    constraints.maxHeight,
                  ),
                ),
              );
            },
          ),
        ],
      ),
    );
  }
}

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Size serverImageSize;
  final Size displaySize;

  DetectionPainter({
    required this.detections,
    required this.serverImageSize,
    required this.displaySize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final double imgW = serverImageSize.width;
    final double imgH = serverImageSize.height;

    // math.max(cover) 대신 math.min(contain)을 사용하여 비율 유지
    final scale = math.min(displaySize.width / imgW, displaySize.height / imgH);

    final offsetX = (displaySize.width - scale * imgW) / 2;
    final offsetY = (displaySize.height - scale * imgH) / 2;

    final boxPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;

    final bgPaint = Paint()..color = Colors.greenAccent.withOpacity(0.75);

    for (final det in detections) {
      final x1 = offsetX + det.box[0] * scale;
      final y1 = offsetY + det.box[1] * scale;
      final x2 = offsetX + det.box[2] * scale;
      final y2 = offsetY + det.box[3] * scale;

      canvas.drawRect(Rect.fromLTRB(x1, y1, x2, y2), boxPaint);

      final label = '${det.name} ${(det.score * 100).toStringAsFixed(0)}%';
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.black,
            fontSize: 11,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      double labelX = x1;
      double labelY = y1 - tp.height - 4;

      if (labelY < 0) {
        labelY = math.max(0.0, y1 + 4);
      }

      if (labelX < 0) {
        labelX = 0;
      }

      final double labelWidth = tp.width + 6;
      if (labelX + labelWidth > displaySize.width) {
        labelX = displaySize.width - labelWidth;
      }

      final tagRect = Rect.fromLTWH(labelX, labelY, labelWidth, tp.height + 4);
      canvas.drawRect(tagRect, bgPaint);
      tp.paint(canvas, Offset(labelX + 3, labelY + 2));
    }
  }

  @override
  bool shouldRepaint(DetectionPainter old) => detections != old.detections;
}
