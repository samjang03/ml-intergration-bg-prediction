// lib/core/ml/config/ml_config.dart

/// Конфігурація ML модуля
class MLConfig {
  // Параметри моделі
  static const String modelFileName = 'glucose_prediction_model.onnx';
  static const String modelConfigFileName = 'model_config.json';
  static const int predictionHorizonMinutes = 60;
  static const int requiredHistoryHours = 6;
  
  // Параметри валідації
  static const double minGlucoseValue = 1.0; // ммоль/л
  static const double maxGlucoseValue = 30.0; // ммоль/л
  static const double maxGlucoseRateChange = 0.5; // ммоль/л за 5 хв
  static const double minConfidenceThreshold = 0.3;
  
  // Параметри кешування
  static const Duration featureCacheDuration = Duration(minutes: 2);
  static const Duration predictionCacheDuration = Duration(minutes: 5);
  static const int maxCacheSize = 100;
  
  // Діапазони глікемії (ммоль/л)
  static const double severeHypoThreshold = 3.0;
  static const double hypoThreshold = 3.9;
  static const double targetRangeMin = 3.9;
  static const double targetRangeMax = 10.0;
  static const double hyperThreshold = 10.0;
  static const double severeHyperThreshold = 13.9;
  
  // Вагові коефіцієнти з навченої моделі
  static const double hypoWeight = 2.43;
  static const double hyperWeight = 1.01;
  
  // Параметри оптимізації
  static const bool enableFeatureCaching = true;
  static const bool enablePredictionCaching = true;
  static const bool enableParallelProcessing = false; // Поки що відключено
  
  // Налагодження та логування
  static const bool enableDebugLogging = true;
  static const bool enablePerformanceMetrics = true;
  
  /// Отримання повного шляху до файлу моделі
  static String getModelPath() => 'assets/ml/models/$modelFileName';
  
  /// Отримання повного шляху до конфіга моделі
  static String getConfigPath() => 'assets/ml/config/$modelConfigFileName';
  
  /// Перевірка, чи значення глюкози в допустимих межах
  static bool isValidGlucoseValue(double value) {
    return value >= minGlucoseValue && value <= maxGlucoseValue && value.isFinite;
  }
  
  /// Отримання категорії глікемії за значенням
  static GlycemiaCategory getGlycemiaCategory(double glucose) {
    if (glucose < severeHypoThreshold) {
      return GlycemiaCategory.severehypoglycemia;
    } else if (glucose < hypoThreshold) {
      return GlycemiaCategory.hypoglycemia;
    } else if (glucose <= targetRangeMax) {
      return GlycemiaCategory.targetRange;
    } else if (glucose <= severeHyperThreshold) {
      return GlycemiaCategory.hyperglycemia;
    } else {
      return GlycemiaCategory.severeHyperglycemia;
    }
  }
}

// lib/core/ml/interfaces/i_prediction_service.dart

import '../models/prediction_context.dart';
import '../models/prediction_result.dart';

/// Інтерфейс для сервісу прогнозування глюкози
abstract class IPredictionService {
  /// Створення прогнозу рівня глюкози
  Future<PredictionResult> predict(PredictionContext context);
  
  /// Пакетне прогнозування для кількох контекстів
  Future<List<PredictionResult>> predictBatch(List<PredictionContext> contexts);
  
  /// Перевірка готовності сервісу до роботи
  Future<bool> isReady();
  
  /// Ініціалізація сервісу
  Future<void> initialize();
  
  /// Очищення кешу та ресурсів
  Future<void> dispose();
}

// lib/core/ml/interfaces/i_feature_service.dart

import '../models/prediction_context.dart';
import '../models/feature_vector.dart';

/// Інтерфейс для сервісу підготовки ознак
abstract class IFeatureService {
  /// Підготовка ознак для прогнозування
  Future<FeatureVector> prepareFeatures(PredictionContext context);
  
  /// Отримання списку імен ознак
  List<String> getFeatureNames();
  
  /// Валідація контексту
  bool validateContext(PredictionContext context);
  
  /// Очищення кешу ознак
  void clearCache();
}

// lib/core/ml/services/ml_service_locator.dart

import 'package:get_it/get_it.dart';
import '../interfaces/i_prediction_service.dart';
import '../interfaces/i_feature_service.dart';
import 'feature_engineering_service.dart';
import 'prediction_service.dart';
import 'model_loader_service.dart';

/// Локатор сервісів для ML модуля
class MLServiceLocator {
  static final GetIt _getIt = GetIt.instance;
  
  /// Ініціалізація всіх ML сервісів
  static Future<void> initialize() async {
    // Реєстрація сервісів
    _getIt.registerSingleton<IFeatureService>(FeatureEngineeringService());
    _getIt.registerSingleton<ModelLoaderService>(ModelLoaderService());
    
    // Створення і реєстрація сервісу прогнозування
    final predictionService = PredictionService(
      featureService: _getIt<IFeatureService>(),
      modelLoader: _getIt<ModelLoaderService>(),
    );
    
    _getIt.registerSingleton<IPredictionService>(predictionService);
    
    // Ініціалізація сервісів
    await _getIt<ModelLoaderService>().initialize();
    await _getIt<IPredictionService>().initialize();
  }
  
  /// Отримання сервісу прогнозування
  static IPredictionService get predictionService => _getIt<IPredictionService>();
  
  /// Отримання сервісу підготовки ознак
  static IFeatureService get featureService => _getIt<IFeatureService>();
  
  /// Отримання завантажувача моделей
  static ModelLoaderService get modelLoader => _getIt<ModelLoaderService>();
  
  /// Очищення всіх сервісів
  static Future<void> dispose() async {
    await _getIt<IPredictionService>().dispose();
    await _getIt.reset();
  }
}

// lib/core/ml/services/model_loader_service.dart

import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import '../config/ml_config.dart';
import '../exceptions/ml_exceptions.dart';

/// Сервіс для завантаження ML моделі та її метаданих
class ModelLoaderService {
  Uint8List? _modelData;
  Map<String, dynamic>? _modelConfig;
  bool _isInitialized = false;
  
  /// Чи ініціалізований сервіс
  bool get isInitialized => _isInitialized;
  
  /// Дані моделі
  Uint8List? get modelData => _modelData;
  
  /// Конфігурація моделі
  Map<String, dynamic>? get modelConfig => _modelConfig;
  
  /// Ініціалізація завантажувача
  Future<void> initialize() async {
    try {
      debugPrint('MLModelLoader: Ініціалізація...');
      
      // Завантаження моделі
      await _loadModel();
      
      // Завантаження конфігурації
      await _loadConfig();
      
      _isInitialized = true;
      debugPrint('MLModelLoader: Успішно ініціалізовано');
      
    } catch (e) {
      throw ModelLoadException.withTimestamp(
        'Помилка ініціалізації завантажувача моделі',
        details: e.toString(),
      );
    }
  }
  
  /// Завантаження файлу моделі
  Future<void> _loadModel() async {
    try {
      final modelPath = MLConfig.getModelPath();
      debugPrint('MLModelLoader: Завантаження моделі з $modelPath');
      
      final ByteData data = await rootBundle.load(modelPath);
      _modelData = data.buffer.asUint8List();
      
      debugPrint('MLModelLoader: Модель завантажена (${_modelData!.length} байт)');
      
    } catch (e) {
      throw ModelLoadException.withTimestamp(
        'Не вдалося завантажити файл моделі',
        details: e.toString(),
      );
    }
  }
  
  /// Завантаження конфігурації моделі
  Future<void> _loadConfig() async {
    try {
      final configPath = MLConfig.getConfigPath();
      debugPrint('MLModelLoader: Завантаження конфігурації з $configPath');
      
      final String configString = await rootBundle.loadString(configPath);
      _modelConfig = json.decode(configString) as Map<String, dynamic>;
      
      // Валідація конфігурації
      _validateConfig();
      
      debugPrint('MLModelLoader: Конфігурація завантажена');
      
    } catch (e) {
      throw ModelLoadException.withTimestamp(
        'Не вдалося завантажити конфігурацію моделі',
        details: e.toString(),
      );
    }
  }
  
  /// Валідація конфігурації моделі
  void _validateConfig() {
    if (_modelConfig == null) {
      throw ModelLoadException('Конфігурація моделі порожня');
    }
    
    final requiredKeys = [
      'model_version',
      'feature_count',
      'input_features',
      'model_params',
    ];
    
    for (final key in requiredKeys) {
      if (!_modelConfig!.containsKey(key)) {
        throw ModelLoadException('Відсутній обов\'язковий ключ в конфігурації: $key');
      }
    }
    
    // Перевірка кількості ознак
    final featureCount = _modelConfig!['feature_count'] as int;
    final inputFeatures = _modelConfig!['input_features'] as List;
    
    if (inputFeatures.length != featureCount) {
      throw ModelLoadException(
        'Невідповідність кількості ознак: очікувалось $featureCount, отримано ${inputFeatures.length}'
      );
    }
  }
  
  /// Отримання імен ознак з конфігурації
  List<String> getFeatureNames() {
    if (!_isInitialized || _modelConfig == null) {
      throw StateError('Завантажувач моделі не ініціалізований');
    }
    
    final features = _modelConfig!['input_features'] as List;
    return features.cast<String>();
  }
  
  /// Отримання версії моделі
  String getModelVersion() {
    if (!_isInitialized || _modelConfig == null) {
      throw StateError('Завантажувач моделі не ініціалізований');
    }
    
    return _modelConfig!['model_version'] as String;
  }
  
  /// Отримання параметрів моделі
  Map<String, dynamic> getModelParams() {
    if (!_isInitialized || _modelConfig == null) {
      throw StateError('Завантажувач моделі не ініціалізований');
    }
    
    return _modelConfig!['model_params'] as Map<String, dynamic>;
  }
  
  /// Перезавантаження моделі
  Future<void> reload() async {
    _isInitialized = false;
    _modelData = null;
    _modelConfig = null;
    
    await initialize();
  }
}

// lib/core/ml/services/ml_cache_service.dart

import 'dart:collection';
import '../models/feature_vector.dart';
import '../models/prediction_result.dart';
import '../config/ml_config.dart';

/// Сервіс кешування для ML операцій
class MLCacheService {
  // Кеш для векторів ознак
  final _featureCache = LinkedHashMap<String, _CacheItem<FeatureVector>>();
  
  // Кеш для результатів прогнозування
  final _predictionCache = LinkedHashMap<String, _CacheItem<PredictionResult>>();
  
  /// Збереження вектора ознак в кеш
  void cacheFeatureVector(String key, FeatureVector vector) {
    if (!MLConfig.enableFeatureCaching) return;
    
    _cleanCache(_featureCache, MLConfig.featureCacheDuration);
    
    if (_featureCache.length >= MLConfig.maxCacheSize) {
      _featureCache.remove(_featureCache.keys.first);
    }
    
    _featureCache[key] = _CacheItem(vector, DateTime.now());
  }
  
  /// Отримання вектора ознак з кешу
  FeatureVector? getCachedFeatureVector(String key) {
    if (!MLConfig.enableFeatureCaching) return null;
    
    final item = _featureCache[key];
    if (item == null) return null;
    
    final age = DateTime.now().difference(item.timestamp);
    if (age > MLConfig.featureCacheDuration) {
      _featureCache.remove(key);
      return null;
    }
    
    return item.data;
  }
  
  /// Збереження результату прогнозування в кеш
  void cachePredictionResult(String key, PredictionResult result) {
    if (!MLConfig.enablePredictionCaching) return;
    
    _cleanCache(_predictionCache, MLConfig.predictionCacheDuration);
    
    if (_predictionCache.length >= MLConfig.maxCacheSize) {
      _predictionCache.remove(_predictionCache.keys.first);
    }
    
    _predictionCache[key] = _CacheItem(result, DateTime.now());
  }
  
  /// Отримання результату прогнозування з кешу
  PredictionResult? getCachedPredictionResult(String key) {
    if (!MLConfig.enablePredictionCaching) return null;
    
    final item = _predictionCache[key];
    if (item == null) return null;
    
    final age = DateTime.now().difference(item.timestamp);
    if (age > MLConfig.predictionCacheDuration) {
      _predictionCache.remove(key);
      return null;
    }
    
    return item.data;
  }
  
  /// Очищення застарілих записів з кешу
  void _cleanCache<T>(LinkedHashMap<String, _CacheItem<T>> cache, Duration maxAge) {
    final now = DateTime.now();
    final keysToRemove = <String>[];
    
    for (final entry in cache.entries) {
      final age = now.difference(entry.value.timestamp);
      if (age > maxAge) {
        keysToRemove.add(entry.key);
      }
    }
    
    for (final key in keysToRemove) {
      cache.remove(key);
    }
  }
  
  /// Очищення всіх кешів
  void clearAll() {
    _featureCache.clear();
    _predictionCache.clear();
  }
  
  /// Статистика кешування
  Map<String, int> getCacheStats() {
    return {
      'feature_cache_size': _featureCache.length,
      'prediction_cache_size': _predictionCache.length,
      'total_cache_size': _featureCache.length + _predictionCache.length,
    };
  }
}

/// Елемент кешу з часовою міткою
class _CacheItem<T> {
  final T data;
  final DateTime timestamp;
  
  const _CacheItem(this.data, this.timestamp);
}

// lib/core/ml/utils/ml_performance_monitor.dart

import 'package:flutter/foundation.dart';
import '../config/ml_config.dart';

/// Монітор продуктивності ML операцій
class MLPerformanceMonitor {
  static final Map<String, List<Duration>> _operationTimes = {};
  static final Map<String, int> _operationCounts = {};
  
  /// Початок вимірювання операції
  static Stopwatch startOperation(String operationName) {
    if (!MLConfig.enablePerformanceMetrics) {
      return Stopwatch(); // Повертаємо порожній секундомір
    }
    
    final stopwatch = Stopwatch()..start();
    return stopwatch;
  }
  
  /// Завершення вимірювання операції
  static void endOperation(String operationName, Stopwatch stopwatch) {
    if (!MLConfig.enablePerformanceMetrics) return;
    
    stopwatch.stop();
    final duration = stopwatch.elapsed;
    
    _operationTimes.putIfAbsent(operationName, () => <Duration>[]);
    _operationTimes[operationName]!.add(duration);
    
    _operationCounts[operationName] = (_operationCounts[operationName] ?? 0) + 1;
    
    if (MLConfig.enableDebugLogging) {
      debugPrint('ML Performance: $operationName took ${duration.inMilliseconds}ms');
    }
  }
  
  /// Отримання статистики продуктивності
  static Map<String, Map<String, dynamic>> getPerformanceStats() {
    final stats = <String, Map<String, dynamic>>{};
    
    for (final operation in _operationTimes.keys) {
      final times = _operationTimes[operation]!;
      final count = _operationCounts[operation]!;
      
      if (times.isNotEmpty) {
        final totalMs = times.fold<int>(0, (sum, duration) => sum + duration.inMilliseconds);
        final avgMs = totalMs / times.length;
        final minMs = times.map((d) => d.inMilliseconds).reduce((a, b) => a < b ? a : b);
        final maxMs = times.map((d) => d.inMilliseconds).reduce((a, b) => a > b ? a : b);
        
        stats[operation] = {
          'count': count,
          'total_ms': totalMs,
          'average_ms': avgMs.round(),
          'min_ms': minMs,
          'max_ms': maxMs,
        };
      }
    }
    
    return stats;
  }
  
  /// Очищення статистики
  static void clearStats() {
    _operationTimes.clear();
    _operationCounts.clear();
  }
  
  /// Логування поточної статистики
  static void logStats() {
    if (!MLConfig.enableDebugLogging) return;
    
    final stats = getPerformanceStats();
    debugPrint('=== ML Performance Statistics ===');
    
    for (final entry in stats.entries) {
      final op = entry.key;
      final data = entry.value;
      debugPrint('$op: ${data['count']} calls, avg: ${data['average_ms']}ms, min: ${data['min_ms']}ms, max: ${data['max_ms']}ms');
    }
    
    debugPrint('=================================');
  }
}
