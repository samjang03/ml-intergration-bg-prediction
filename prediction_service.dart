// lib/core/ml/services/prediction_service.dart

import 'package:flutter/foundation.dart';
import '../interfaces/i_prediction_service.dart';
import '../interfaces/i_feature_service.dart';
import '../models/prediction_context.dart';
import '../models/prediction_result.dart';
import '../models/feature_vector.dart';
import '../config/ml_config.dart';
import '../exceptions/ml_exceptions.dart';
import '../utils/feature_validator.dart';
import '../utils/ml_performance_monitor.dart';
import 'model_loader_service.dart';
import 'ml_cache_service.dart';

/// Основний сервіс прогнозування рівня глюкози
class PredictionService implements IPredictionService {
  final IFeatureService _featureService;
  final ModelLoaderService _modelLoader;
  final MLCacheService _cacheService;
  
  bool _isInitialized = false;
  
  // Тут буде інтеграція з ONNX Runtime або TensorFlow Lite
  // Поки що використовуємо заглушку
  dynamic _mlModel;
  
  PredictionService({
    required IFeatureService featureService,
    required ModelLoaderService modelLoader,
    MLCacheService? cacheService,
  }) : _featureService = featureService,
       _modelLoader = modelLoader,
       _cacheService = cacheService ?? MLCacheService();

  @override
  Future<void> initialize() async {
    final stopwatch = MLPerformanceMonitor.startOperation('service_initialization');
    
    try {
      debugPrint('PredictionService: Ініціалізація...');
      
      // Перевірка готовності залежностей
      if (!_modelLoader.isInitialized) {
        throw PredictionException('ModelLoader не ініціалізований');
      }
      
      // Завантаження ML моделі
      await _loadMLModel();
      
      // Валідація сумісності моделі та сервісу ознак
      await _validateModelCompatibility();
      
      _isInitialized = true;
      debugPrint('PredictionService: Успішно ініціалізовано');
      
    } catch (e) {
      throw PredictionException.withTimestamp(
        'Помилка ініціалізації сервісу прогнозування',
        details: e.toString(),
      );
    } finally {
      MLPerformanceMonitor.endOperation('service_initialization', stopwatch);
    }
  }

  @override
  Future<bool> isReady() async {
    return _isInitialized && 
           _modelLoader.isInitialized && 
           _mlModel != null;
  }

  @override
  Future<PredictionResult> predict(PredictionContext context) async {
    if (!await isReady()) {
      throw PredictionException('Сервіс прогнозування не готовий до роботи');
    }
    
    final stopwatch = MLPerformanceMonitor.startOperation('prediction');
    
    try {
      // Генерація ключа для кешування
      final cacheKey = _generateCacheKey(context);
      
      // Перевірка кешу
      final cachedResult = _cacheService.getCachedPredictionResult(cacheKey);
      if (cachedResult != null) {
        debugPrint('PredictionService: Використано кешований результат');
        return cachedResult;
      }
      
      // Валідація контексту
      final validation = FeatureValidator.validateContext(context);
      if (!validation.isValid) {
        throw DataValidationException(
          'Невалідний контекст прогнозування: ${validation.errors.join(', ')}'
        );
      }
      
      // Логування попереджень
      if (validation.hasWarnings && MLConfig.enableDebugLogging) {
        debugPrint('PredictionService: Попередження: ${validation.warnings.join(', ')}');
      }
      
      // Підготовка ознак
      final featureVector = await _prepareFeatures(context, cacheKey);
      
      // Прогнозування
      final prediction = await _performPrediction(featureVector);
      
      // Валідація результату
      final resultValidation = FeatureValidator.validatePredictionResult(prediction);
      if (!resultValidation.isValid) {
        throw PredictionException(
          'Невалідний результат прогнозування: ${resultValidation.errors.join(', ')}'
        );
      }
      
      // Розрахунок рівня впевненості
      final confidence = _calculateConfidence(context, featureVector, prediction);
      
      // Створення результату
      final result = PredictionResult(
        predictedValue: prediction,
        confidenceLevel: confidence,
        predictionTime: context.predictionTime,
        targetTime: context.predictionTime.add(
          const Duration(minutes: MLConfig.predictionHorizonMinutes)
        ),
        usedFeatures: featureVector,
        isValid: true,
        metadata: {
          'model_version': _modelLoader.getModelVersion(),
          'feature_count': featureVector.length,
          'cache_used': false,
        },
      );
      
      // Збереження в кеш
      _cacheService.cachePredictionResult(cacheKey, result);
      
      if (MLConfig.enableDebugLogging) {
        debugPrint('PredictionService: Прогноз ${prediction.toStringAsFixed(1)} ммоль/л '
                  '(впевненість: ${(confidence * 100).toStringAsFixed(1)}%)');
      }
      
      return result;
      
    } catch (e) {
      debugPrint('PredictionService: Помилка прогнозування: $e');
      
      return PredictionResult.error(
        error: e.toString(),
        predictionTime: context.predictionTime,
        targetTime: context.predictionTime.add(
          const Duration(minutes: MLConfig.predictionHorizonMinutes)
        ),
        usedFeatures: FeatureVector.invalid(
          error: 'Не вдалося підготувати ознаки',
          predictionTime: context.predictionTime,
        ),
      );
    } finally {
      MLPerformanceMonitor.endOperation('prediction', stopwatch);
    }
  }

  @override
  Future<List<PredictionResult>> predictBatch(List<PredictionContext> contexts) async {
    final results = <PredictionResult>[];
    
    for (final context in contexts) {
      try {
        final result = await predict(context);
        results.add(result);
      } catch (e) {
        // Для пакетної обробки не зупиняємось на помилках
        results.add(PredictionResult.error(
          error: e.toString(),
          predictionTime: context.predictionTime,
          targetTime: context.predictionTime.add(
            const Duration(minutes: MLConfig.predictionHorizonMinutes)
          ),
          usedFeatures: FeatureVector.invalid(
            error: 'Помилка пакетної обробки',
            predictionTime: context.predictionTime,
          ),
        ));
      }
    }
    
    return results;
  }

  @override
  Future<void> dispose() async {
    debugPrint('PredictionService: Звільнення ресурсів...');
    
    _cacheService.clearAll();
    _isInitialized = false;
    _mlModel = null;
    
    if (MLConfig.enablePerformanceMetrics) {
      MLPerformanceMonitor.logStats();
    }
  }

  // Приватні методи

  Future<void> _loadMLModel() async {
    final stopwatch = MLPerformanceMonitor.startOperation('model_loading');
    
    try {
      // TODO: Тут буде інтеграція з ONNX Runtime або TensorFlow Lite
      // Поки що створюємо заглушку
      _mlModel = _MockMLModel(_modelLoader.modelData!);
      
      debugPrint('PredictionService: ML модель завантажена');
      
    } catch (e) {
      throw ModelLoadException.withTimestamp(
        'Не вдалося завантажити ML модель',
        details: e.toString(),
      );
    } finally {
      MLPerformanceMonitor.endOperation('model_loading', stopwatch);
    }
  }

  Future<void> _validateModelCompatibility() async {
    try {
      final modelFeatures = _modelLoader.getFeatureNames();
      final serviceFeatures = _featureService.getFeatureNames();
      
      if (modelFeatures.length != serviceFeatures.length) {
        throw ModelLoadException(
          'Невідповідність кількості ознак: модель очікує ${modelFeatures.length}, '
          'сервіс генерує ${serviceFeatures.length}'
        );
      }
      
      // Перевірка відповідності імен ознак
      for (int i = 0; i < modelFeatures.length; i++) {
        if (modelFeatures[i] != serviceFeatures[i]) {
          throw ModelLoadException(
            'Невідповідність ознак на позиції $i: '
            'модель очікує "${modelFeatures[i]}", сервіс генерує "${serviceFeatures[i]}"'
          );
        }
      }
      
      debugPrint('PredictionService: Валідація сумісності пройшла успішно');
      
    } catch (e) {
      throw PredictionException.withTimestamp(
        'Помилка валідації сумісності моделі',
        details: e.toString(),
      );
    }
  }

  Future<FeatureVector> _prepareFeatures(PredictionContext context, String cacheKey) async {
    final stopwatch = MLPerformanceMonitor.startOperation('feature_preparation');
    
    try {
      // Перевірка кешу ознак
      final cachedFeatures = _cacheService.getCachedFeatureVector(cacheKey + '_features');
      if (cachedFeatures != null) {
        debugPrint('PredictionService: Використано кешовані ознаки');
        return cachedFeatures;
      }
      
      // Підготовка ознак
      final featureVector = await _featureService.prepareFeatures(context);
      
      if (!featureVector.isValid) {
        throw FeatureEngineeringException(
          'Не вдалося підготувати ознаки: ${featureVector.error}'
        );
      }
      
      // Збереження в кеш
      _cacheService.cacheFeatureVector(cacheKey + '_features', featureVector);
      
      return featureVector;
      
    } finally {
      MLPerformanceMonitor.endOperation('feature_preparation', stopwatch);
    }
  }

  Future<double> _performPrediction(FeatureVector featureVector) async {
    final stopwatch = MLPerformanceMonitor.startOperation('ml_inference');
    
    try {
      // Підготовка даних для моделі
      final modelFeatures = _modelLoader.getFeatureNames();
      final inputData = featureVector.toList(modelFeatures);
      
      // Виконання прогнозування
      // TODO: Замінити на реальний виклик ONNX/TensorFlow
      final prediction = await _mlModel.predict(inputData);
      
      // Перевірка результату
      if (!prediction.isFinite || prediction < 0) {
        throw PredictionException('Модель повернула невалідний результат: $prediction');
      }
      
      return prediction;
      
    } finally {
      MLPerformanceMonitor.endOperation('ml_inference', stopwatch);
    }
  }

  double _calculateConfidence(PredictionContext context, FeatureVector features, double prediction) {
    // Базовий рівень впевненості
    double confidence = 0.7;
    
    // Бонус за кількість історичних даних
    final historyHours = context.glucoseHistoryHours;
    if (historyHours >= 6.0) {
      confidence += 0.2;
    } else if (historyHours >= 3.0) {
      confidence += 0.1;
    }
    
    // Бонус за свіжість даних
    final latestGlucose = context.glucoseHistory
        .map((r) => r.timestamp)
        .reduce((a, b) => a.isAfter(b) ? a : b);
    final dataAge = context.predictionTime.difference(latestGlucose).inMinutes;
    
    if (dataAge <= 5) {
      confidence += 0.1;
    } else if (dataAge <= 10) {
      confidence += 0.05;
    }
    
    // Зниження впевненості для екстремальних значень
    if (prediction < 3.0 || prediction > 15.0) {
      confidence *= 0.8;
    }
    
    // Зниження впевненості при швидкій зміні глюкози
    final glucoseRate = features.features['glucose_rate'] ?? 0.0;
    if (glucoseRate.abs() > 0.3) {
      confidence *= 0.9;
    }
    
    return confidence.clamp(0.0, 1.0);
  }

  String _generateCacheKey(PredictionContext context) {
    // Створюємо ключ на основі хешу основних параметрів
    final keyComponents = [
      context.predictionTime.millisecondsSinceEpoch ~/ (5 * 60 * 1000), // Округлення до 5 хвилин
      context.glucoseHistory.length,
      context.insulinHistory.length,
      context.carbHistory.length,
      context.activityHistory.length,
    ];
    
    if (context.glucoseHistory.isNotEmpty) {
      keyComponents.add(context.glucoseHistory.last.timestamp.millisecondsSinceEpoch);
      keyComponents.add((context.glucoseHistory.last.mmolL * 100).round());
    }
    
    return keyComponents.join('_');
  }
}

/// Заглушка ML моделі для тестування
/// TODO: Замінити на реальну інтеграцію з ONNX Runtime
class _MockMLModel {
  final Uint8List _modelData;
  
  _MockMLModel(this._modelData);
  
  Future<double> predict(List<double> features) async {
    // Імітація часу виконання моделі
    await Future.delayed(const Duration(milliseconds: 50));
    
    // Спрощена логіка для демонстрації
    // Базується на поточному рівні глюкози та швидкості зміни
    double currentGlucose = features[0]; // bg-0-00
    double glucoseRate = features.length > 288 ? features[288] : 0.0; // glucose_rate
    double activeInsulin = features.length > 289 ? features[289] : 0.0;
    double activeCarbs = features.length > 290 ? features[290] : 0.0;
    
    // Спрощений розрахунок прогнозу
    double prediction = currentGlucose;
    
    // Вплив тренду
    prediction += glucoseRate * 12; // 12 інтервалів по 5 хвилин = 60 хвилин
    
    // Вплив активного інсуліну (знижує глюкозу)
    prediction -= activeInsulin * 0.5;
    
    // Вплив активних вуглеводів (підвищує глюкозу)
    prediction += activeCarbs * 0.1;
    
    // Додавання невеликого шуму для реалістичності
    final random = DateTime.now().millisecondsSinceEpoch % 100;
    prediction += (random - 50) * 0.01;
    
    // Обмеження фізіологічними межами
    return prediction.clamp(2.0, 25.0);
  }
}
