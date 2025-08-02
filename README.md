# Fever Prediction System

An intelligent web application that uses machine learning to predict fever types based on symptoms. This system can classify between Normal, Dengue, Malaria, and Typhoid fever conditions using an ensemble of machine learning models.

## Features

- **AI-Powered Diagnosis**: Advanced ensemble model combining Random Forest and XGBoost algorithms
- **Real-time Analysis**: Instant fever type prediction with confidence scores
- **User-Friendly Interface**: Modern, responsive web interface with intuitive symptom selection
- **Model Explainability**: SHAP analysis for understanding feature importance
- **Medical Disclaimer**: Professional medical disclaimers and privacy protection
- **Secure & Private**: No data storage - all analysis is performed locally

## Supported Fever Types

The system can predict the following fever conditions:
- **Normal**: No fever or mild temperature variations
- **Dengue**: Viral fever transmitted by mosquitoes
- **Malaria**: Parasitic disease transmitted by infected mosquitoes  
- **Typhoid**: Bacterial infection caused by Salmonella typhi

## Input Parameters

The model analyzes the following symptoms:
- **Headache**: Presence or absence of head pain
- **Chills**: Shivering or feeling cold
- **Body Pain**: Muscle aches and joint pain
- **Fatigue**: Unusual tiredness or weakness
- **Rash**: Skin irritation or rash presence
- **Temperature**: Body temperature in Fahrenheit

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fever_Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if needed)
   ```bash
   python train_model.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
Fever_Prediction/
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── shap_analysis.py       # Model explainability analysis
├── dataset.csv           # Training dataset
├── requirements.txt      # Python dependencies
├── model/
│   └── ensemble_model.pkl # Trained ensemble model
├── templates/
│   ├── index.html        # Main input form
│   └── result.html       # Results display page
└── static/
    └── styles.css        # CSS styling
```

## Model Architecture

### Ensemble Model
The system uses a voting classifier that combines:
- **Random Forest Classifier**: 100 estimators with random state 42
- **XGBoost Classifier**: Gradient boosting with default parameters
- **LightGBM Classifier**: Light gradient boosting machine

### Model Performance
- **Accuracy**: High accuracy on test dataset
- **Voting Method**: Soft voting for probability-based predictions
- **Feature Importance**: SHAP analysis for model interpretability

## Usage

1. **Access the Application**: Open the web interface at `http://localhost:5000`

2. **Input Symptoms**: 
   - Select symptom presence/absence for each category
   - Enter current body temperature in Fahrenheit
   - All fields are required for accurate prediction

3. **Get Results**: 
   - View predicted fever type
   - See confidence score percentage
   - Review medical disclaimer

## Model Analysis

### SHAP Analysis
Run the SHAP analysis to understand feature importance:
```bash
python shap_analysis.py
```

This generates visualizations showing:
- Feature importance rankings
- Individual prediction explanations
- Model interpretability insights

## Dataset Information

The training dataset contains labeled examples with the following features:
- **Symptom indicators**: Binary values (0/1) for presence/absence
- **Temperature**: Continuous values in Fahrenheit
- **Target labels**: Fever type classification

## Security & Privacy

- **No Data Storage**: User inputs are processed in memory only
- **Local Processing**: All analysis performed on the server
- **Medical Disclaimer**: Clear disclaimers about medical advice
- **Secure Interface**: Professional-grade security measures

## Medical Disclaimer

**Important**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for proper medical care.

## Technical Requirements

### Dependencies
- Flask 2.0+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+
- xgboost 1.5+
- lightgbm 3.3+
- shap 0.40+
- matplotlib 3.5+

### System Requirements
- Python 3.7+
- 4GB RAM (recommended)
- Web browser with JavaScript enabled

## Development

### Adding New Features
1. Modify the Flask routes in `app.py`
2. Update the HTML templates in `templates/`
3. Enhance the model in `train_model.py`
4. Test thoroughly before deployment

### Model Retraining
To retrain the model with new data:
1. Update `dataset.csv` with new labeled examples
2. Run `python train_model.py`
3. The new model will be saved to `model/ensemble_model.pkl`

## License

This project is for educational purposes. Please ensure compliance with local regulations regarding medical software and data privacy.

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass before submitting
- Medical accuracy is maintained
- Privacy and security standards are upheld

## Support

For questions or issues:
1. Check the documentation
2. Review the code comments
3. Ensure all dependencies are installed
4. Verify the dataset format

---

**Note**: This is a demonstration project and should not be used for actual medical diagnosis without proper validation and regulatory approval. 
