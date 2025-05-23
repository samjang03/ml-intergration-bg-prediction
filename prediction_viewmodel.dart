import 'package:flutter/foundation.dart';
import '../../../core/ml/models/prediction_result.dart';
import '../services/glucose_prediction_service.dart';

/// ViewModel для UI компонента прогнозування
class PredictionViewModel extends ChangeNotifier {
  final GlucosePredictionService _predictionService;
  
  PredictionResult? _currentPrediction;
  bool _isLoading = false;
  String? _error;
  
  PredictionViewModel({
    required GlucosePredictionService predictionService,
  }) : _predictionService = predictionService;

  // Getters
  PredictionResult? get currentPrediction => _currentPrediction;
  bool get isLoading => _isLoading;
  String? get error => _error;
  bool get hasPrediction => _currentPrediction?.isValid == true;

  /// Оновлення прогнозу
  Future<void> updatePrediction(String userId) async {
    _setLoading(true);
    _setError(null);
    
    try {
      final result = await _predictionService.createPrediction(userId: userId);
      _currentPrediction = result;
      
      if (!result.isValid) {
        _setError('Не вдалося створити надійний прогноз: ${result.error}');
      }
      
    } catch (e) {
      _setError('Помилка отримання прогнозу: $e');
      _currentPrediction = null;
    } finally {
      _setLoading(false);
    }
  }

  /// Автоматичне оновлення прогнозу кожні 5 хвилин
  void startAutoUpdate(String userId) {
    Timer.periodic(const Duration(minutes: 5), (timer) async {
      if (!_isLoading) {
        await updatePrediction(userId);
      }
    });
  }

  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void _setError(String? error) {
    _error = error;
    notifyListeners();
  }
}
