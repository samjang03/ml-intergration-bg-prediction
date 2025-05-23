// lib/core/ml/models/prediction_context.dart

import '../../../data/models/glucose_reading.dart';
import '../../../data/models/insulin_record.dart';
import '../../../data/models/carb_record.dart';
import '../../../data/models/activity_record.dart';

/// Контекст для прогнозування, що містить всі необхідні дані
class PredictionContext {
  final List<GlucoseReading> glucoseHistory;
  final List<InsulinRecord> insulinHistory;
  final List<CarbRecord> carbHistory;
  final List<ActivityRecord> activityHistory;
  final DateTime predictionTime;

  const PredictionContext({
    required this.glucoseHistory,
    required this.insulinHistory,
    required this.carbHistory,
    required this.activityHistory,
    required this.predictionTime,
  });

  /// Створення контексту з поточного стану бази даних
  factory PredictionContext.fromCurrentState({
    required List<GlucoseReading> allGlucoseReadings,
    required List<InsulinRecord> allInsulinRecords,
    required List<CarbRecord> allCarbRecords,
    required List<ActivityRecord> allActivityRecords,
    DateTime? predictionTime,
  }) {
    final now = predictionTime ?? DateTime.now();
    final sixHoursAgo = now.subtract(const Duration(hours: 6));

    return PredictionContext(
      glucoseHistory: allGlucoseReadings
          .where((reading) => reading.timestamp.isAfter(sixHoursAgo))
          .toList(),
      insulinHistory: allInsulinRecords
          .where((record) => record.timestamp.isAfter(sixHoursAgo))
          .toList(),
      carbHistory: allCarbRecords
          .where((record) => record.timestamp.isAfter(sixHoursAgo))
          .toList(),
      activityHistory: allActivityRecords
          .where((record) => record.timestamp.isAfter(sixHoursAgo))
          .toList(),
      predictionTime: now,
    );
  }

  /// Валідація контексту
  bool get isValid {
    return glucoseHistory.isNotEmpty && 
           predictionTime.isBefore(DateTime.now().add(const Duration(minutes: 5)));
  }

  /// Кількість годин історії глюкози
  double get glucoseHistoryHours {
    if (glucoseHistory.isEmpty) return 0.0;
    
    final oldest = glucoseHistory
        .map((r) => r.timestamp)
        .reduce((a, b) => a.isBefore(b) ? a : b);
    
    return predictionTime.difference(oldest).inMinutes / 60.0;
  }

  @override
  String toString() {
    return 'PredictionContext('
           'glucose: ${glucoseHistory.length} readings, '
           'insulin: ${insulinHistory.length} records, '
           'carbs: ${carbHistory.length} records, '
           'activity: ${activityHistory.length} records, '
           'time: $predictionTime)';
  }
}

// lib/core/ml/models/feature_vector.dart

/// Вектор ознак для ML моделі
class FeatureVector {
  final Map<String, double> features;
  final DateTime createdAt;
  final DateTime predictionTime;
  final bool isValid;
  final String? error;

  const FeatureVector({
    required this.features,
    required this.createdAt,
    required this.predictionTime,
    required this.isValid,
    this.error,
  });

  /// Створення неvalidного вектора з помилкою
  factory FeatureVector.invalid({
    required String error,
    required DateTime predictionTime,
  }) {
    return FeatureVector(
      features: const {},
      createdAt: DateTime.now(),
      predictionTime: predictionTime,
      isValid: false,
      error: error,
    );
  }

  /// Конвертація у список значень у правильному порядку
  List<double> toList(List<String> featureNames) {
    if (!isValid) {
      throw StateError('Cannot convert invalid feature vector to list');
    }

    return featureNames.map((name) => features[name] ?? 0.0).toList();
  }

  /// Кількість ознак
  int get length => features.length;

  /// Чи містить всі необхідні ознаки
  bool hasAllFeatures(List<String> requiredFeatures) {
    return requiredFeatures.every((feature) => features.containsKey(feature));
  }

  @override
  String toString() {
    if (!isValid) {
      return 'FeatureVector(invalid: $error)';
    }
    return 'FeatureVector(${features.length} features, created: $createdAt)';
  }
}

// lib/core/ml/utils/activity_encoder.dart

/// Кодувальник для типів активності відповідно до Python моделі
class ActivityEncoder {
  // Mapping відповідно до LabelEncoder з Python коду
  static const Map<String, int> _activityMapping = {
    'None': 0,
    'Sleeping': 1,
    'Sitting': 2,
    'Standing': 3,
    'Walking': 4,
    'Running': 5,
    'Cycling': 6,
    'Exercise': 7,
    'Sports': 8,
    'Housework': 9,
    'Work': 10,
    'Driving': 11,
    'Eating': 12,
    'Cooking': 13,
    'Shopping': 14,
    'Other': 15,
  };

  static const Map<int, String> _reverseMapping = {
    0: 'None',
    1: 'Sleeping',
    2: 'Sitting',
    3: 'Standing',
    4: 'Walking',
    5: 'Running',
    6: 'Cycling',
    7: 'Exercise',
    8: 'Sports',
    9: 'Housework',
    10: 'Work',
    11: 'Driving',
    12: 'Eating',
    13: 'Cooking',
    14: 'Shopping',
    15: 'Other',
  };

  /// Кодування активності у числове значення
  int encode(String activity) {
    return _activityMapping[activity] ?? _activityMapping['None']!;
  }

  /// Декодування числового значення у назву активності
  String decode(int code) {
    return _reverseMapping[code] ?? 'None';
  }

  /// Список всіх доступних активностей
  List<String> get availableActivities => _activityMapping.keys.toList();

  /// Перевірка, чи підтримується активність
  bool isSupported(String activity) {
    return _activityMapping.containsKey(activity);
  }
}

// lib/core/ml/models/prediction_result.dart

import 'feature_vector.dart';

/// Результат прогнозування рівня глюкози
class PredictionResult {
  final double predictedValue;
  final double confidenceLevel;
  final DateTime predictionTime;
  final DateTime targetTime;
  final FeatureVector usedFeatures;
  final bool isValid;
  final String? error;
  final Map<String, dynamic>? metadata;

  const PredictionResult({
    required this.predictedValue,
    required this.confidenceLevel,
    required this.predictionTime,
    required this.targetTime,
    required this.usedFeatures,
    required this.isValid,
    this.error,
    this.metadata,
  });

  /// Створення результату з помилкою
  factory PredictionResult.error({
    required String error,
    required DateTime predictionTime,
    required DateTime targetTime,
    required FeatureVector usedFeatures,
  }) {
    return PredictionResult(
      predictedValue: 0.0,
      confidenceLevel: 0.0,
      predictionTime: predictionTime,
      targetTime: targetTime,
      usedFeatures: usedFeatures,
      isValid: false,
      error: error,
    );
  }

  /// Горизонт прогнозування у хвилинах
  int get predictionHorizonMinutes => 
      targetTime.difference(predictionTime).inMinutes;

  /// Чи знаходиться прогноз у цільовому діапазоні (3.9-10.0 ммоль/л)
  bool get isInTargetRange => 
      predictedValue >= 3.9 && predictedValue <= 10.0;

  /// Чи є ризик гіпоглікемії
  bool get isHypoglycemiaRisk => predictedValue < 3.9;

  /// Чи є ризик гіперглікемії
  bool get isHyperglycemiaRisk => predictedValue > 10.0;

  /// Рівень ризику (0-1, де 1 - найвищий ризик)
  double get riskLevel {
    if (predictedValue < 3.0) {
      return 1.0; // Критична гіпоглікемія
    } else if (predictedValue < 3.9) {
      return 0.8; // Гіпоглікемія
    } else if (predictedValue > 13.9) {
      return 0.9; // Критична гіперглікемія
    } else if (predictedValue > 10.0) {
      return 0.6; // Гіперглікемія
    } else {
      return 0.0; // Цільовий діапазон
    }
  }

  /// Категорія глікемії
  GlycemiaCategory get category {
    if (predictedValue < 3.0) {
      return GlycemiaCategory.severehypoglycemia;
    } else if (predictedValue < 3.9) {
      return GlycemiaCategory.hypoglycemia;
    } else if (predictedValue <= 10.0) {
      return GlycemiaCategory.targetRange;
    } else if (predictedValue <= 13.9) {
      return GlycemiaCategory.hyperglycemia;
    } else {
      return GlycemiaCategory.severeHyperglycemia;
    }
  }

  @override
  String toString() {
    if (!isValid) {
      return 'PredictionResult(error: $error)';
    }
    return 'PredictionResult('
           'value: ${predictedValue.toStringAsFixed(1)} mmol/L, '
           'confidence: ${(confidenceLevel * 100).toStringAsFixed(1)}%, '
           'horizon: ${predictionHorizonMinutes}min, '
           'category: ${category.name})';
  }
}

/// Категорії глікемії
enum GlycemiaCategory {
  severehypoglycemia('Тяжка гіпоглікемія', 'severe_hypo'),
  hypoglycemia('Гіпоглікемія', 'hypo'),
  targetRange('Цільовий діапазон', 'target'),
  hyperglycemia('Гіперглікемія', 'hyper'),
  severeHyperglycemia('Тяжка гіперглікемія', 'severe_hyper');

  const GlycemiaCategory(this.displayName, this.code);
  
  final String displayName;
  final String code;
}

// lib/core/ml/exceptions/ml_exceptions.dart

/// Базовий клас для ML помилок
abstract class MLException implements Exception {
  final String message;
  final String? details;
  final DateTime timestamp;

  const MLException(this.message, {this.details}) : timestamp = null;

  MLException.withTimestamp(this.message, {this.details}) 
      : timestamp = DateTime.now();

  @override
  String toString() => 'MLException: $message${details != null ? ' ($details)' : ''}';
}

/// Помилка підготовки ознак
class FeatureEngineeringException extends MLException {
  const FeatureEngineeringException(super.message, {super.details});
  
  FeatureEngineeringException.withTimestamp(super.message, {super.details}) 
      : super.withTimestamp();
}

/// Помилка прогнозування
class PredictionException extends MLException {
  const PredictionException(super.message, {super.details});
  
  PredictionException.withTimestamp(super.message, {super.details}) 
      : super.withTimestamp();
}

/// Помилка завантаження моделі
class ModelLoadException extends MLException {
  const ModelLoadException(super.message, {super.details});
  
  ModelLoadException.withTimestamp(super.message, {super.details}) 
      : super.withTimestamp();
}

/// Помилка валідації даних
class DataValidationException extends MLException {
  const DataValidationException(super.message, {super.details});
  
  DataValidationException.withTimestamp(super.message, {super.details}) 
      : super.withTimestamp();
}

// lib/core/ml/utils/feature_validator.dart

import '../models/feature_vector.dart';
import '../models/prediction_context.dart';
import '../exceptions/ml_exceptions.dart';

/// Валідатор для перевірки якості та повноти ознак
class FeatureValidator {
  static const int _expectedFeatureCount = 289; // Відповідно до Python моделі
  static const double _maxGlucoseValue = 30.0; // ммоль/л
  static const double _minGlucoseValue = 1.0; // ммоль/л
  static const double _maxInsulinValue = 50.0; // одиниць
  static const double _maxCarbValue = 200.0; // грамів

  /// Валідація контексту прогнозування
  static ValidationResult validateContext(PredictionContext context) {
    final errors = <String>[];
    final warnings = <String>[];

    // Перевірка наявності історії глюкози
    if (context.glucoseHistory.isEmpty) {
      errors.add('Відсутня історія вимірювань глюкози');
    } else {
      // Перевірка якості даних глюкози
      final invalidGlucose = context.glucoseHistory.where(
        (reading) => reading.mmolL < _minGlucoseValue || reading.mmolL > _maxGlucoseValue
      );
      
      if (invalidGlucose.isNotEmpty) {
        warnings.add('Виявлено ${invalidGlucose.length} невалідних вимірювань глюкози');
      }

      // Перевірка достатності історії
      if (context.glucoseHistoryHours < 6.0) {
        warnings.add('Недостатня історія глюкози: ${context.glucoseHistoryHours.toStringAsFixed(1)} годин');
      }
    }

    // Перевірка даних інсуліну
    final invalidInsulin = context.insulinHistory.where(
      (record) => record.value < 0 || record.value > _maxInsulinValue
    );
    
    if (invalidInsulin.isNotEmpty) {
      warnings.add('Виявлено ${invalidInsulin.length} невалідних записів інсуліну');
    }

    // Перевірка даних вуглеводів
    final invalidCarbs = context.carbHistory.where(
      (record) => record.value < 0 || record.value > _maxCarbValue
    );
    
    if (invalidCarbs.isNotEmpty) {
      warnings.add('Виявлено ${invalidCarbs.length} невалідних записів вуглеводів');
    }

    // Перевірка часу прогнозування
    final now = DateTime.now();
    if (context.predictionTime.isAfter(now.add(const Duration(minutes: 5)))) {
      errors.add('Час прогнозування занадто далеко в майбутньому');
    }

    return ValidationResult(
      isValid: errors.isEmpty,
      errors: errors,
      warnings: warnings,
    );
  }

  /// Валідація вектора ознак
  static ValidationResult validateFeatureVector(FeatureVector vector) {
    final errors = <String>[];
    final warnings = <String>[];

    if (!vector.isValid) {
      errors.add('Вектор ознак позначений як невалідний: ${vector.error}');
      return ValidationResult(isValid: false, errors: errors, warnings: warnings);
    }

    // Перевірка кількості ознак
    if (vector.length != _expectedFeatureCount) {
      warnings.add('Неочікувана кількість ознак: ${vector.length}, очікувалось: $_expectedFeatureCount');
    }

    // Перевірка на NaN та нескінченні значення
    final invalidFeatures = <String>[];
    for (final entry in vector.features.entries) {
      if (!entry.value.isFinite) {
        invalidFeatures.add(entry.key);
      }
    }

    if (invalidFeatures.isNotEmpty) {
      errors.add('Виявлено невалідні значення в ознаках: ${invalidFeatures.join(', ')}');
    }

    // Перевірка критичних ознак
    final criticalFeatures = ['bg-0-00', 'glucose_rate', 'estimated_active_insulin'];
    final missingCritical = criticalFeatures.where(
      (feature) => !vector.features.containsKey(feature)
    ).toList();

    if (missingCritical.isNotEmpty) {
      errors.add('Відсутні критичні ознаки: ${missingCritical.join(', ')}');
    }

    return ValidationResult(
      isValid: errors.isEmpty,
      errors: errors,
      warnings: warnings,
    );
  }

  /// Валідація результату прогнозування
  static ValidationResult validatePredictionResult(double prediction) {
    final errors = <String>[];
    final warnings = <String>[];

    // Перевірка на валідність числа
    if (!prediction.isFinite) {
      errors.add('Прогноз містить невалідне значення: $prediction');
      return ValidationResult(isValid: false, errors: errors, warnings: warnings);
    }

    // Перевірка фізіологічних меж
    if (prediction < 0.5 || prediction > 35.0) {
      errors.add('Прогноз поза фізіологічними межами: ${prediction.toStringAsFixed(1)} ммоль/л');
    }

    // Попередження для екстремальних значень
    if (prediction < 2.0) {
      warnings.add('Прогноз вказує на критично низький рівень глюкози');
    } else if (prediction > 20.0) {
      warnings.add('Прогноз вказує на критично високий рівень глюкози');
    }

    return ValidationResult(
      isValid: errors.isEmpty,
      errors: errors,
      warnings: warnings,
    );
  }
}

/// Результат валідації
class ValidationResult {
  final bool isValid;
  final List<String> errors;
  final List<String> warnings;

  const ValidationResult({
    required this.isValid,
    required this.errors,
    required this.warnings,
  });

  /// Чи є попередження
  bool get hasWarnings => warnings.isNotEmpty;

  /// Загальна кількість проблем
  int get totalIssues => errors.length + warnings.length;

  @override
  String toString() {
    final buffer = StringBuffer();
    buffer.writeln('ValidationResult(valid: $isValid)');
    
    if (errors.isNotEmpty) {
      buffer.writeln('Errors:');
      for (final error in errors) {
        buffer.writeln('  - $error');
      }
    }
    
    if (warnings.isNotEmpty) {
      buffer.writeln('Warnings:');
      for (final warning in warnings) {
        buffer.writeln('  - $warning');
      }
    }
    
    return buffer.toString();
  }
}
