import 'services/ml_service_locator.dart';
import 'interfaces/i_prediction_service.dart';
import 'interfaces/i_feature_service.dart';
import 'config/ml_config.dart';

/// Основний ML модуль для інтеграції з рештою застосунку
class MLModule {
  static bool _isInitialized = false;
  
  /// Ініціалізація ML модуля
  static Future<void> initialize() async {
    if (_isInitialized) return;
    
    try {
      await MLServiceLocator.initialize();
      _isInitialized = true;
      
      print('✅ ML Module успішно ініціалізовано');
    } catch (e) {
      print('❌ Помилка ініціалізації ML Module: $e');
      rethrow;
    }
  }
  
  /// Сервіс прогнозування
  static IPredictionService get predictionService {
    _ensureInitialized();
    return MLServiceLocator.predictionService;
  }
  
  /// Сервіс підготовки ознак
  static IFeatureService get featureService {
    _ensureInitialized();
    return MLServiceLocator.featureService;
  }
  
  /// Чи ініціалізований модуль
  static bool get isInitialized => _isInitialized;
  
  /// Очищення ресурсів
  static Future<void> dispose() async {
    if (!_isInitialized) return;
    
    await MLServiceLocator.dispose();
    _isInitialized = false;
    
    print('🧹 ML Module ресурси звільнено');
  }
  
  static void _ensureInitialized() {
    if (!_isInitialized) {
      throw StateError('ML Module не ініціалізований. Викличте MLModule.initialize() спочатку.');
    }
  }
}
