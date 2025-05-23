// test/core/ml/services/feature_engineering_service_test.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:glucose_app/core/ml/services/feature_engineering_service.dart';
import 'package:glucose_app/core/ml/models/prediction_context.dart';
import 'package:glucose_app/data/models/glucose_reading.dart';
import 'package:glucose_app/data/models/insulin_record.dart';
import 'package:glucose_app/data/models/carb_record.dart';
import 'package:glucose_app/data/models/activity_record.dart';

void main() {
  group('FeatureEngineeringService', () {
    late FeatureEngineeringService service;
    late DateTime baseTime;

    setUp(() {
      service = FeatureEngineeringService();
      baseTime = DateTime(2024, 1, 15, 10, 30); // Базовий час для тестів
    });

    group('Feature Preparation', () {
      test('should create complete feature vector with sufficient data', () async {
        // Arrange
        final context = _createCompleteContext(baseTime);

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isTrue);
        expect(result.features.length, greaterThan(280)); // Очікуємо ~289 ознак
        
        // Перевірка наявності ключових ознак
        expect(result.features.containsKey('bg-0-00'), isTrue);
        expect(result.features.containsKey('glucose_rate'), isTrue);
        expect(result.features.containsKey('estimated_active_insulin'), isTrue);
        expect(result.features.containsKey('estimated_active_carbs'), isTrue);
        expect(result.features.containsKey('sin_hour'), isTrue);
        expect(result.features.containsKey('cos_hour'), isTrue);
      });

      test('should handle minimal glucose data', () async {
        // Arrange
        final context = _createMinimalContext(baseTime);

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isTrue);
        expect(result.features['bg-0-00'], equals(8.5)); // Поточне значення
        expect(result.features['glucose_rate'], equals(0.0)); // Недостатньо даних для швидкості
      });

      test('should calculate glucose rate correctly', () async {
        // Arrange
        final glucoseHistory = [
          GlucoseReading(
            readingId: 1,
            userId: 'test',
            timestamp: baseTime,
            value: 153.0, // 8.5 ммоль/л
            mmolL: 8.5,
            trendDescription: 'stable',
          ),
          GlucoseReading(
            readingId: 2,
            userId: 'test',
            timestamp: baseTime.subtract(const Duration(minutes: 5)),
            value: 144.0, // 8.0 ммоль/л
            mmolL: 8.0,
            trendDescription: 'stable',
          ),
        ];

        final context = PredictionContext(
          glucoseHistory: glucoseHistory,
          insulinHistory: [],
          carbHistory: [],
          activityHistory: [],
          predictionTime: baseTime,
        );

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isTrue);
        expect(result.features['glucose_rate'], equals(0.5)); // (8.5 - 8.0) / 1 = 0.5 ммоль/л за 5 хв
      });

      test('should calculate active insulin correctly', () async {
        // Arrange
        final insulinHistory = [
          InsulinRecord(
            recordId: 1,
            userId: 'test',
            timestamp: baseTime.subtract(const Duration(minutes: 30)), // 0.5 годин тому
            value: 5.0,
            type: 'bolus',
          ),
          InsulinRecord(
            recordId: 2,
            userId: 'test',
            timestamp: baseTime.subtract(const Duration(hours: 2)), // 2 години тому
            value: 3.0,
            type: 'bolus',
          ),
        ];

        final context = PredictionContext(
          glucoseHistory: [_createGlucoseReading(baseTime, 8.5)],
          insulinHistory: insulinHistory,
          carbHistory: [],
          activityHistory: [],
          predictionTime: baseTime,
        );

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isTrue);
        // Очікуємо: 5.0 * 0.85 + 3.0 * 0.6 = 4.25 + 1.8 = 6.05
        expect(result.features['estimated_active_insulin'], closeTo(6.05, 0.1));
      });

      test('should calculate active carbs correctly', () async {
        // Arrange
        final carbHistory = [
          CarbRecord(
            recordId: 1,
            userId: 'test',
            timestamp: baseTime.subtract(const Duration(minutes: 30)), // 0.5 годин тому
            value: 50,
            mealType: 'lunch',
          ),
          CarbRecord(
            recordId: 2,
            userId: 'test',
            timestamp: baseTime.subtract(const Duration(hours: 3)), // 3 години тому (поза межами)
            value: 30,
            mealType: 'breakfast',
          ),
        ];

        final context = PredictionContext(
          glucoseHistory: [_createGlucoseReading(baseTime, 8.5)],
          insulinHistory: [],
          carbHistory: carbHistory,
          activityHistory: [],
          predictionTime: baseTime,
        );

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isTrue);
        // Очікуємо: 50 * 0.75 = 37.5 (тільки перший запис в межах 2 годин)
        expect(result.features['estimated_active_carbs'], closeTo(37.5, 0.1));
      });

      test('should create cyclic time features correctly', () async {
        // Arrange - 10:30 AM
        final context = _createMinimalContext(baseTime);

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isTrue);
        
        // Перевірка циклічних ознак
        final expectedSinHour = math.sin(2 * math.pi * 10 / 24);
        final expectedCosHour = math.cos(2 * math.pi * 10 / 24);
        final expectedSinMinute = math.sin(2 * math.pi * 30 / 60);
        final expectedCosMinute = math.cos(2 * math.pi * 30 / 60);

        expect(result.features['sin_hour'], closeTo(expectedSinHour, 0.001));
        expect(result.features['cos_hour'], closeTo(expectedCosHour, 0.001));
        expect(result.features['sin_minute'], closeTo(expectedSinMinute, 0.001));
        expect(result.features['cos_minute'], closeTo(expectedCosMinute, 0.001));
      });

      test('should encode activities correctly', () async {
        // Arrange
        final activityHistory = [
          ActivityRecord(
            recordId: 1,
            userId: 'test',
            timestamp: baseTime.subtract(const Duration(minutes: 10)),
            activityType: 'Walking',
          ),
          ActivityRecord(
            recordId: 2,
            userId: 'test',
            timestamp: baseTime.subtract(const Duration(minutes: 20)),
            activityType: 'Sitting',
          ),
        ];

        final context = PredictionContext(
          glucoseHistory: [_createGlucoseReading(baseTime, 8.5)],
          insulinHistory: [],
          carbHistory: [],
          activityHistory: activityHistory,
          predictionTime: baseTime,
        );

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isTrue);
        
        // Перевірка кодування активності
        // 'Walking' має код 4, 'Sitting' - код 2
        expect(result.features.containsKey('activity-0-10'), isTrue);
        expect(result.features['activity-0-10'], equals(4.0)); // Walking
      });
    });

    group('Error Handling', () {
      test('should handle empty glucose history', () async {
        // Arrange
        final context = PredictionContext(
          glucoseHistory: [],
          insulinHistory: [],
          carbHistory: [],
          activityHistory: [],
          predictionTime: baseTime,
        );

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isFalse);
        expect(result.error, contains('Glucose history cannot be empty'));
      });

      test('should handle invalid glucose values', () async {
        // Arrange
        final glucoseHistory = [
          GlucoseReading(
            readingId: 1,
            userId: 'test',
            timestamp: baseTime,
            value: double.nan,
            mmolL: double.nan,
            trendDescription: 'stable',
          ),
        ];

        final context = PredictionContext(
          glucoseHistory: glucoseHistory,
          insulinHistory: [],
          carbHistory: [],
          activityHistory: [],
          predictionTime: baseTime,
        );

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        // Сервіс має впоратися з NaN значеннями
        expect(result.isValid, isTrue);
        expect(result.features['bg-0-00']?.isFinite, isTrue);
      });

      test('should clamp extreme glucose rate values', () async {
        // Arrange - створюємо нереально швидку зміну глюкози
        final glucoseHistory = [
          GlucoseReading(
            readingId: 1,
            userId: 'test',
            timestamp: baseTime,
            value: 360.0, // 20.0 ммоль/л
            mmolL: 20.0,
            trendDescription: 'rising',
          ),
          GlucoseReading(
            readingId: 2,
            userId: 'test',
            timestamp: baseTime.subtract(const Duration(minutes: 5)),
            value: 72.0, // 4.0 ммоль/л
            mmolL: 4.0,
            trendDescription: 'stable',
          ),
        ];

        final context = PredictionContext(
          glucoseHistory: glucoseHistory,
          insulinHistory: [],
          carbHistory: [],
          activityHistory: [],
          predictionTime: baseTime,
        );

        // Act
        final result = await service.prepareFeatures(context);

        // Assert
        expect(result.isValid, isTrue);
        // Швидкість має бути обмежена до 0.5 ммоль/л за 5 хв
        expect(result.features['glucose_rate'], equals(0.5));
      });
    });

    group('Feature Names', () {
      test('should return correct feature names in proper order', () {
        // Act
        final featureNames = service.getFeatureNames();

        // Assert
        expect(featureNames.length, equals(289)); // Загальна кількість ознак
        
        // Перевіряємо перші кілька ознак
        expect(featureNames[0], equals('bg-0-00'));
        expect(featureNames[1], equals('bg-0-05'));
        expect(featureNames[12], equals('bg-1-00'));
        
        // Перевіряємо останні ознаки
        expect(featureNames.contains('glucose_rate'), isTrue);
        expect(featureNames.contains('estimated_active_insulin'), isTrue);
        expect(featureNames.contains('sin_hour'), isTrue);
        expect(featureNames.contains('cos_hour'), isTrue);
      });
    });
  });
}

// Допоміжні методи для створення тестових даних

PredictionContext _createCompleteContext(DateTime baseTime) {
  final glucoseHistory = <GlucoseReading>[];
  final insulinHistory = <InsulinRecord>[];
  final carbHistory = <CarbRecord>[];
  final activityHistory = <ActivityRecord>[];

  // Створюємо 6 годин історії глюкози (кожні 5 хвилин)
  for (int minutes = 0; minutes < 360; minutes += 5) {
    final timestamp = baseTime.subtract(Duration(minutes: minutes));
    final value = 8.0 + (math.sin(minutes / 60.0) * 2.0); // Симуляція коливань
    
    glucoseHistory.add(GlucoseReading(
      readingId: minutes ~/ 5,
      userId: 'test',
      timestamp: timestamp,
      value: value * 18.0, // Конвертація в мг/дл
      mmolL: value,
      trendDescription: 'stable',
    ));
  }

  // Кілька доз інсуліну
  insulinHistory.addAll([
    InsulinRecord(
      recordId: 1,
      userId: 'test',
      timestamp: baseTime.subtract(const Duration(minutes: 30)),
      value: 4.0,
      type: 'bolus',
    ),
    InsulinRecord(
      recordId: 2,
      userId: 'test',
      timestamp: baseTime.subtract(const Duration(hours: 2)),
      value: 2.0,
      type: 'bolus',
    ),
  ]);

  // Кілька прийомів їжі
  carbHistory.addAll([
    CarbRecord(
      recordId: 1,
      userId: 'test',
      timestamp: baseTime.subtract(const Duration(minutes: 45)),
      value: 60,
      mealType: 'lunch',
    ),
    CarbRecord(
      recordId: 2,
      userId: 'test',
      timestamp: baseTime.subtract(const Duration(hours: 3)),
      value: 45,
      mealType: 'breakfast',
    ),
  ]);

  // Кілька активностей
  activityHistory.addAll([
    ActivityRecord(
      recordId: 1,
      userId: 'test',
      timestamp: baseTime.subtract(const Duration(minutes: 15)),
      activityType: 'Sitting',
    ),
    ActivityRecord(
      recordId: 2,
      userId: 'test',
      timestamp: baseTime.subtract(const Duration(hours: 1)),
      activityType: 'Walking',
    ),
  ]);

  return PredictionContext(
    glucoseHistory: glucoseHistory,
    insulinHistory: insulinHistory,
    carbHistory: carbHistory,
    activityHistory: activityHistory,
    predictionTime: baseTime,
  );
}

PredictionContext _createMinimalContext(DateTime baseTime) {
  final glucoseHistory = [
    _createGlucoseReading(baseTime, 8.5),
  ];

  return PredictionContext(
    glucoseHistory: glucoseHistory,
    insulinHistory: [],
    carbHistory: [],
    activityHistory: [],
    predictionTime: baseTime,
  );
}

GlucoseReading _createGlucoseReading(DateTime timestamp, double mmolL) {
  return GlucoseReading(
    readingId: timestamp.millisecondsSinceEpoch,
    userId: 'test',
    timestamp: timestamp,
    value: mmolL * 18.0, // Конвертація в мг/дл
    mmolL: mmolL,
    trendDescription: 'stable',
  );
}
