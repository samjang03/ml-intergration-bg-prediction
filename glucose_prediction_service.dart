import '../../../core/ml/ml_module.dart';
import '../../../core/ml/models/prediction_context.dart';
import '../../../core/ml/models/prediction_result.dart';
import '../../../data/repositories/glucose_repository.dart';
import '../../../data/repositories/insulin_repository.dart';
import '../../../data/repositories/carb_repository.dart';
import '../../../data/repositories/activity_repository.dart';

/// Сервіс для інтеграції ML прогнозування з бізнес-логікою застосунку
class GlucosePredictionService {
  final GlucoseRepository _glucoseRepository;
  final InsulinRepository _insulinRepository;
  final CarbRepository _carbRepository;
  final ActivityRepository _activityRepository;
  
  GlucosePredictionService({
    required GlucoseRepository glucoseRepository,
    required InsulinRepository insulinRepository,
    required CarbRepository carbRepository,
    required ActivityRepository activityRepository,
  }) : _glucoseRepository = glucoseRepository,
       _insulinRepository = insulinRepository,
       _carbRepository = carbRepository,
       _activityRepository = activityRepository;

  /// Створення прогнозу для поточного користувача
  Future<PredictionResult> createPrediction({
    required String userId,
    DateTime? predictionTime,
  }) async {
    try {
      // Збір всіх необхідних даних з репозиторіїв
      final context = await _buildPredictionContext(userId, predictionTime);
      
      // Виконання прогнозування через ML модуль
      final result = await MLModule.predictionService.predict(context);
      
      // Можна додати додаткову бізнес-логіку обробки результату
      await _processPredictionResult(userId, result);
      
      return result;
      
    } catch (e) {
      print('Помилка створення прогнозу для користувача $userId: $e');
      rethrow;
    }
  }

  /// Створення множинних прогнозів (наприклад, для графіка)
  Future<List<PredictionResult>> createMultiplePredictions({
    required String userId,
    required List<DateTime> predictionTimes,
  }) async {
    final contexts = <PredictionContext>[];
    
    for (final time in predictionTimes) {
      final context = await _buildPredictionContext(userId, time);
      contexts.add(context);
    }
    
    return await MLModule.predictionService.predictBatch(contexts);
  }

  /// Побудова контексту прогнозування з даних користувача
  Future<PredictionContext> _buildPredictionContext(
    String userId, 
    DateTime? predictionTime,
  ) async {
    final now = predictionTime ?? DateTime.now();
    final sixHoursAgo = now.subtract(const Duration(hours: 6));
    
    // Паралельне завантаження всіх типів даних
    final results = await Future.wait([
      _glucoseRepository.getReadingsInRange(userId, sixHoursAgo, now),
      _insulinRepository.getRecordsInRange(userId, sixHoursAgo, now),
      _carbRepository.getRecordsInRange(userId, sixHoursAgo, now),
      _activityRepository.getRecordsInRange(userId, sixHoursAgo, now),
    ]);
    
    return PredictionContext(
      glucoseHistory: results[0] as List<GlucoseReading>,
      insulinHistory: results[1] as List<InsulinRecord>,
      carbHistory: results[2] as List<CarbRecord>,
      activityHistory: results[3] as List<ActivityRecord>,
      predictionTime: now,
    );
  }

  /// Обробка результату прогнозування (наприклад, збереження, сповіщення)
  Future<void> _processPredictionResult(String userId, PredictionResult result) async {
    // Тут можна додати:
    // - Збереження результату в базу даних
    // - Генерацію сповіщень для критичних прогнозів
    // - Логування для аналітики
    // - Відправлення даних на сервер
    
    if (result.isValid && result.isHypoglycemiaRisk) {
      print('⚠️ Прогноз вказує на ризик гіпоглікемії: ${result.predictedValue.toStringAsFixed(1)} ммоль/л');
      // TODO: Генерація сповіщення
    }
    
    if (result.isValid && result.isHyperglycemiaRisk) {
      print('⚠️ Прогноз вказує на ризик гіперглікемії: ${result.predictedValue.toStringAsFixed(1)} ммоль/л');
      // TODO: Генерація сповіщення
    }
  }
}
