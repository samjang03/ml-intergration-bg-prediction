import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import '../../../data/models/glucose_reading.dart';
import '../../../data/models/insulin_record.dart';
import '../../../data/models/carb_record.dart';
import '../../../data/models/activity_record.dart';
import '../models/prediction_context.dart';
import '../models/feature_vector.dart';
import '../utils/activity_encoder.dart';

/// Сервіс для підготовки ознак (Feature Engineering) для ML моделі прогнозування глюкози
/// 
/// Відповідає за:
/// - Створення часових ознак
/// - Розрахунок швидкості зміни глюкози
/// - Обчислення активного інсуліну (IOB)
/// - Обчислення активних вуглеводів (COB)
/// - Кодування активності
/// - Підготовку повного вектора ознак для моделі
class FeatureEngineeringService {
  static const int _requiredGlucoseHistoryHours = 6;
  static const int _measurementsPerHour = 12; // кожні 5 хвилин
  static const int _totalGlucoseFeatures = 73; // 6 годин × 12 + поточне значення
  static const int _totalInsulinFeatures = 72; // 6 годин × 12
  static const int _totalCarbFeatures = 72; // 6 годин × 12
  static const int _totalActivityFeatures = 72; // 6 годин × 12

  final ActivityEncoder _activityEncoder;

  FeatureEngineeringService() : _activityEncoder = ActivityEncoder();

  /// Підготовка повного вектора ознак для ML моделі
  Future<FeatureVector> prepareFeatures(PredictionContext context) async {
    try {
      // Валідація вхідних даних
      _validateInput(context);

      final features = <String, double>{};

      // 1. Історичні значення глюкози (bg-X-XX)
      final glucoseFeatures = _createGlucoseFeatures(context.glucoseHistory, context.predictionTime);
      features.addAll(glucoseFeatures);

      // 2. Дози інсуліну (insulin-X-XX)
      final insulinFeatures = _createInsulinFeatures(context.insulinHistory, context.predictionTime);
      features.addAll(insulinFeatures);

      // 3. Кількість вуглеводів (carbs-X-XX)
      final carbFeatures = _createCarbFeatures(context.carbHistory, context.predictionTime);
      features.addAll(carbFeatures);

      // 4. Активність (activity-X-XX)
      final activityFeatures = _createActivityFeatures(context.activityHistory, context.predictionTime);
      features.addAll(activityFeatures);

      // 5. Часові ознаки (циклічні)
      final timeFeatures = _createTimeFeatures(context.predictionTime);
      features.addAll(timeFeatures);

      // 6. Швидкість зміни глюкози
      final glucoseRate = _calculateGlucoseRate(context.glucoseHistory);
      features['glucose_rate'] = glucoseRate;

      // 7. Оцінка активного інсуліну (IOB)
      final activeInsulin = _calculateActiveInsulin(context.insulinHistory, context.predictionTime);
      features['estimated_active_insulin'] = activeInsulin;

      // 8. Оцінка активних вуглеводів (COB)
      final activeCarbs = _calculateActiveCarbs(context.carbHistory, context.predictionTime);
      features['estimated_active_carbs'] = activeCarbs;

      // 9. Співвідношення інсулін/вуглеводи
      final insulinCarbRatio = _calculateInsulinCarbRatio(activeInsulin, activeCarbs);
      features['insulin_carb_ratio'] = insulinCarbRatio;

      // Перевірка повноти вектора ознак
      _validateFeatureVector(features);

      return FeatureVector(
        features: features,
        createdAt: DateTime.now(),
        predictionTime: context.predictionTime,
        isValid: true,
      );

    } catch (e) {
      debugPrint('Error in feature engineering: $e');
      return FeatureVector.invalid(
        error: e.toString(),
        predictionTime: context.predictionTime,
      );
    }
  }

  /// Створення ознак історичних значень глюкози
  Map<String, double> _createGlucoseFeatures(List<GlucoseReading> history, DateTime predictionTime) {
    final features = <String, double>{};
    
    // Сортування за часом (найновіші спочатку)
    final sortedHistory = List<GlucoseReading>.from(history)
      ..sort((a, b) => b.timestamp.compareTo(a.timestamp));

    // Генерація ознак bg-H-MM для останніх 6 годин
    for (int hour = 0; hour < _requiredGlucoseHistoryHours; hour++) {
      for (int intervalIndex = 0; intervalIndex < _measurementsPerHour; intervalIndex++) {
        final minutes = intervalIndex * 5;
        final featureName = 'bg-$hour-${minutes.toString().padLeft(2, '0')}';
        
        // Обчислення цільового часу для цієї ознаки
        final targetTime = predictionTime.subtract(
          Duration(hours: hour, minutes: minutes)
        );

        // Пошук найближчого виміру
        final value = _findClosestGlucoseValue(sortedHistory, targetTime);
        features[featureName] = value;
      }
    }

    return features;
  }

  /// Створення ознак доз інсуліну
  Map<String, double> _createInsulinFeatures(List<InsulinRecord> history, DateTime predictionTime) {
    final features = <String, double>{};
    
    for (int hour = 0; hour < _requiredGlucoseHistoryHours; hour++) {
      for (int intervalIndex = 0; intervalIndex < _measurementsPerHour; intervalIndex++) {
        final minutes = intervalIndex * 5;
        final featureName = 'insulin-$hour-${minutes.toString().padLeft(2, '0')}';
        
        final startTime = predictionTime.subtract(
          Duration(hours: hour, minutes: minutes + 5)
        );
        final endTime = predictionTime.subtract(
          Duration(hours: hour, minutes: minutes)
        );

        // Сума інсуліну за 5-хвилинний інтервал
        final insulinSum = _sumInsulinInInterval(history, startTime, endTime);
        features[featureName] = insulinSum;
      }
    }

    return features;
  }

  /// Створення ознак кількості вуглеводів
  Map<String, double> _createCarbFeatures(List<CarbRecord> history, DateTime predictionTime) {
    final features = <String, double>{};
    
    for (int hour = 0; hour < _requiredGlucoseHistoryHours; hour++) {
      for (int intervalIndex = 0; intervalIndex < _measurementsPerHour; intervalIndex++) {
        final minutes = intervalIndex * 5;
        final featureName = 'carbs-$hour-${minutes.toString().padLeft(2, '0')}';
        
        final startTime = predictionTime.subtract(
          Duration(hours: hour, minutes: minutes + 5)
        );
        final endTime = predictionTime.subtract(
          Duration(hours: hour, minutes: minutes)
        );

        // Сума вуглеводів за 5-хвилинний інтервал
        final carbSum = _sumCarbsInInterval(history, startTime, endTime);
        features[featureName] = carbSum;
      }
    }

    return features;
  }

  /// Створення ознак активності
  Map<String, double> _createActivityFeatures(List<ActivityRecord> history, DateTime predictionTime) {
    final features = <String, double>{};
    
    for (int hour = 0; hour < _requiredGlucoseHistoryHours; hour++) {
      for (int intervalIndex = 0; intervalIndex < _measurementsPerHour; intervalIndex++) {
        final minutes = intervalIndex * 5;
        final featureName = 'activity-$hour-${minutes.toString().padLeft(2, '0')}';
        
        final startTime = predictionTime.subtract(
          Duration(hours: hour, minutes: minutes + 5)
        );
        final endTime = predictionTime.subtract(
          Duration(hours: hour, minutes: minutes)
        );

        // Пошук активності в інтервалі
        final activity = _findActivityInInterval(history, startTime, endTime);
        final encodedActivity = _activityEncoder.encode(activity);
        features[featureName] = encodedActivity.toDouble();
      }
    }

    return features;
  }

  /// Створення циклічних часових ознак
  Map<String, double> _createTimeFeatures(DateTime time) {
    final hour = time.hour;
    final minute = time.minute;

    return {
      'sin_hour': math.sin(2 * math.pi * hour / 24),
      'cos_hour': math.cos(2 * math.pi * hour / 24),
      'sin_minute': math.sin(2 * math.pi * minute / 60),
      'cos_minute': math.cos(2 * math.pi * minute / 60),
    };
  }

  /// Розрахунок швидкості зміни глюкози (ммоль/л за 5 хв)
  double _calculateGlucoseRate(List<GlucoseReading> history) {
    if (history.length < 2) return 0.0;

    final sortedHistory = List<GlucoseReading>.from(history)
      ..sort((a, b) => b.timestamp.compareTo(a.timestamp));

    final current = sortedHistory[0];
    final previous = sortedHistory[1];

    final timeDiffMinutes = current.timestamp.difference(previous.timestamp).inMinutes;
    if (timeDiffMinutes == 0) return 0.0;

    final glucoseDiff = current.mmolL - previous.mmolL;
    final rate = glucoseDiff / (timeDiffMinutes / 5.0); // нормалізація до 5 хв

    // Обмеження фізіологічно можливими значеннями
    return rate.clamp(-0.5, 0.5);
  }

  /// Розрахунок активного інсуліну (IOB) з урахуванням часу дії
  double _calculateActiveInsulin(List<InsulinRecord> history, DateTime currentTime) {
    double activeInsulin = 0.0;

    for (final record in history) {
      final timeDiff = currentTime.difference(record.timestamp);
      final hoursAgo = timeDiff.inMinutes / 60.0;

      // Модель поглинання інсуліну (спрощена)
      double activityFactor = 0.0;
      if (hoursAgo <= 1.0) {
        activityFactor = 0.85; // піковий період
      } else if (hoursAgo <= 2.0) {
        activityFactor = 0.6; // активний період
      } else if (hoursAgo <= 3.0) {
        activityFactor = 0.35; // залишковий період
      } else if (hoursAgo <= 4.0) {
        activityFactor = 0.1; // мінімальний вплив
      }
      // Після 4 годин інсулін вважається неактивним

      activeInsulin += record.value * activityFactor;
    }

    return activeInsulin;
  }

  /// Розрахунок активних вуглеводів (COB)
  double _calculateActiveCarbs(List<CarbRecord> history, DateTime currentTime) {
    double activeCarbs = 0.0;

    for (final record in history) {
      final timeDiff = currentTime.difference(record.timestamp);
      final hoursAgo = timeDiff.inMinutes / 60.0;

      // Вуглеводи активні протягом ~2 годин
      if (hoursAgo <= 2.0) {
        // Лінійне зменшення активності
        final activityFactor = math.max(0.0, 1.0 - (hoursAgo / 2.0));
        activeCarbs += record.value * activityFactor;
      }
    }

    return activeCarbs;
  }

  /// Розрахунок співвідношення інсулін/вуглеводи
  double _calculateInsulinCarbRatio(double activeInsulin, double activeCarbs) {
    const epsilon = 1e-6; // для уникнення ділення на нуль
    return (activeInsulin + epsilon) / (activeCarbs + epsilon);
  }

  // Допоміжні методи

  double _findClosestGlucoseValue(List<GlucoseReading> history, DateTime targetTime) {
    if (history.isEmpty) return 0.0;

    GlucoseReading? closest;
    Duration? minDifference;

    for (final reading in history) {
      final difference = targetTime.difference(reading.timestamp).abs();
      if (minDifference == null || difference < minDifference) {
        minDifference = difference;
        closest = reading;
      }
    }

    // Якщо найближче значення занадто далеко (>15 хвилин), використовуємо інтерполяцію
    if (minDifference != null && minDifference.inMinutes > 15) {
      return _interpolateGlucoseValue(history, targetTime) ?? closest?.mmolL ?? 0.0;
    }

    return closest?.mmolL ?? 0.0;
  }

  double? _interpolateGlucoseValue(List<GlucoseReading> history, DateTime targetTime) {
    if (history.length < 2) return null;

    // Знаходимо два найближчі виміри з різних боків від цільового часу
    GlucoseReading? before;
    GlucoseReading? after;

    for (final reading in history) {
      if (reading.timestamp.isBefore(targetTime)) {
        if (before == null || reading.timestamp.isAfter(before.timestamp)) {
          before = reading;
        }
      } else {
        if (after == null || reading.timestamp.isBefore(after.timestamp)) {
          after = reading;
        }
      }
    }

    if (before == null || after == null) return null;

    // Лінійна інтерполяція
    final totalDuration = after.timestamp.difference(before.timestamp).inMinutes;
    final targetDuration = targetTime.difference(before.timestamp).inMinutes;
    
    if (totalDuration == 0) return before.mmolL;

    final ratio = targetDuration / totalDuration;
    return before.mmolL + (after.mmolL - before.mmolL) * ratio;
  }

  double _sumInsulinInInterval(List<InsulinRecord> history, DateTime start, DateTime end) {
    return history
        .where((record) => 
            record.timestamp.isAfter(start) && 
            record.timestamp.isBefore(end))
        .fold(0.0, (sum, record) => sum + record.value);
  }

  double _sumCarbsInInterval(List<CarbRecord> history, DateTime start, DateTime end) {
    return history
        .where((record) => 
            record.timestamp.isAfter(start) && 
            record.timestamp.isBefore(end))
        .fold(0.0, (sum, record) => sum + record.value.toDouble());
  }

  String _findActivityInInterval(List<ActivityRecord> history, DateTime start, DateTime end) {
    final activitiesInInterval = history
        .where((record) => 
            record.timestamp.isAfter(start) && 
            record.timestamp.isBefore(end))
        .toList();

    if (activitiesInInterval.isEmpty) return 'None';

    // Повертаємо останню активність в інтервалі
    activitiesInInterval.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return activitiesInInterval.first.activityType;
  }

  void _validateInput(PredictionContext context) {
    if (context.glucoseHistory.isEmpty) {
      throw ArgumentError('Glucose history cannot be empty');
    }

    // Перевірка, чи є достатньо історичних даних
    final oldestRequiredTime = context.predictionTime.subtract(
      Duration(hours: _requiredGlucoseHistoryHours)
    );

    final hasEnoughHistory = context.glucoseHistory.any(
      (reading) => reading.timestamp.isBefore(oldestRequiredTime.add(Duration(minutes: 30)))
    );

    if (!hasEnoughHistory) {
      debugPrint('Warning: Insufficient glucose history for optimal prediction');
    }
  }

  void _validateFeatureVector(Map<String, double> features) {
    // Перевірка наявності всіх критичних ознак
    final requiredFeatures = [
      'bg-0-00', // поточне значення глюкози
      'glucose_rate',
      'estimated_active_insulin',
      'estimated_active_carbs',
      'insulin_carb_ratio',
      'sin_hour',
      'cos_hour',
    ];

    for (final feature in requiredFeatures) {
      if (!features.containsKey(feature)) {
        throw StateError('Missing required feature: $feature');
      }
    }

    // Перевірка на NaN або некінцеві значення
    for (final entry in features.entries) {
      if (!entry.value.isFinite) {
        debugPrint('Warning: Invalid value for feature ${entry.key}: ${entry.value}');
        features[entry.key] = 0.0; // Заміна на безпечне значення
      }
    }

    debugPrint('Feature vector validated: ${features.length} features prepared');
  }

  /// Метод для отримання списку всіх ознак у правильному порядку
  List<String> getFeatureNames() {
    final features = <String>[];

    // 1. Glucose features (bg-X-XX)
    for (int hour = 0; hour < _requiredGlucoseHistoryHours; hour++) {
      for (int intervalIndex = 0; intervalIndex < _measurementsPerHour; intervalIndex++) {
        final minutes = intervalIndex * 5;
        features.add('bg-$hour-${minutes.toString().padLeft(2, '0')}');
      }
    }

    // 2. Insulin features (insulin-X-XX)
    for (int hour = 0; hour < _requiredGlucoseHistoryHours; hour++) {
      for (int intervalIndex = 0; intervalIndex < _measurementsPerHour; intervalIndex++) {
        final minutes = intervalIndex * 5;
        features.add('insulin-$hour-${minutes.toString().padLeft(2, '0')}');
      }
    }

    // 3. Carb features (carbs-X-XX)
    for (int hour = 0; hour < _requiredGlucoseHistoryHours; hour++) {
      for (int intervalIndex = 0; intervalIndex < _measurementsPerHour; intervalIndex++) {
        final minutes = intervalIndex * 5;
        features.add('carbs-$hour-${minutes.toString().padLeft(2, '0')}');
      }
    }

    // 4. Activity features (activity-X-XX)
    for (int hour = 0; hour < _requiredGlucoseHistoryHours; hour++) {
      for (int intervalIndex = 0; intervalIndex < _measurementsPerHour; intervalIndex++) {
        final minutes = intervalIndex * 5;
        features.add('activity-$hour-${minutes.toString().padLeft(2, '0')}');
      }
    }

    // 5. Engineered features
    features.addAll([
      'glucose_rate',
      'estimated_active_insulin',
      'estimated_active_carbs',
      'insulin_carb_ratio',
      'sin_hour',
      'cos_hour',
      'sin_minute',
      'cos_minute',
    ]);

    return features;
  }
}
