import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import logging
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ApplicationProcessor:
    """Handles application data validation and preprocessing for Prudential dataset"""
    
    def __init__(self):
        self.required_fields = [
            'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_2',
            'Employment_Info_3', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3',
            'Insurance_History_1', 'Family_Hist_1', 'Medical_History_1', 'Medical_History_2',
            'Medical_History_4'
        ] + [f'Medical_Keyword_{i}' for i in range(1, 49)]
        
    def validate_application(self, application_data):
        """Validate application data for completeness and consistency"""
        missing_fields = [field for field in self.required_fields 
                         if field not in application_data or pd.isna(application_data[field])]
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
            
        # Basic data type and range validation
        try:
            if not (0 <= application_data['Ins_Age'] <= 1):  # Normalized age
                return False, "Normalized age must be between 0 and 1"
            if not (0 <= application_data['BMI'] <= 1):  # Normalized BMI
                return False, "BMI must be between 0 and 1"
            if application_data['Employment_Info_1'] < 0:
                return False, "Employment income cannot be negative"
        except (TypeError, ValueError):
            return False, "Invalid data types in application"
            
        return True, "Application data valid"

    def preprocess_data(self, application_data):
        """Convert application data to numerical format for ML model"""
        processed_data = application_data.copy()
        
        # Handle missing values (impute with median for numerical, mode for categorical)
        numerical_cols = ['Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Family_Hist_1']
        categorical_cols = ['Employment_Info_2', 'Employment_Info_3', 'InsuredInfo_1', 
                          'InsuredInfo_2', 'InsuredInfo_3', 'Insurance_History_1', 
                          'Medical_History_1', 'Medical_History_2', 'Medical_History_4']
        
        for col in numerical_cols:
            if pd.isna(processed_data[col]):
                processed_data[col] = 0.5  # Median imputation for normalized values
        for col in categorical_cols:
            if pd.isna(processed_data[col]):
                processed_data[col] = 1  # Mode imputation for categorical
                
        return processed_data
    """Handles application data validation and preprocessing for Prudential dataset"""
    
    def __init__(self):
        self.required_fields = [
            'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_2',
            'Employment_Info_3', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3',
            'Insurance_History_1', 'Family_Hist_1', 'Medical_History_1', 'Medical_History_2',
            'Medical_History_4'
        ] + [f'Medical_Keyword_{i}' for i in range(1, 49)]
        
    def validate_application(self, application_data):
        """Validate application data for completeness and consistency"""
        missing_fields = [field for field in self.required_fields 
                         if field not in application_data or pd.isna(application_data[field])]
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
            
        # Basic data type and range validation
        try:
            if not (0 <= application_data['Ins_Age'] <= 1):  # Normalized age
                return False, "Normalized age must be between 0 and 1"
            if not (0 <= application_data['BMI'] <= 1):  # Normalized BMI
                return False, "BMI must be between 0 and 1"
            if application_data['Employment_Info_1'] < 0:
                return False, "Employment income cannot be negative"
        except (TypeError, ValueError):
            return False, "Invalid data types in application"
            
        return True, "Application data valid"

    def preprocess_data(self, application_data):
        """Convert application data to numerical format for ML model"""
        processed_data = application_data.copy()
        
        # Handle missing values (impute with median for numerical, mode for categorical)
        numerical_cols = ['Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Family_Hist_1']
        categorical_cols = ['Employment_Info_2', 'Employment_Info_3', 'InsuredInfo_1', 
                          'InsuredInfo_2', 'InsuredInfo_3', 'Insurance_History_1', 
                          'Medical_History_1', 'Medical_History_2', 'Medical_History_4']
        
        for col in numerical_cols:
            if pd.isna(processed_data[col]):
                processed_data[col] = 0.5  # Median imputation for normalized values
        for col in categorical_cols:
            if pd.isna(processed_data[col]):
                processed_data[col] = 1  # Mode imputation for categorical
                
        return processed_data

class RiskEvaluator:
    """Evaluates risk using ML model and Prudential dataset features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_columns = [
            'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_2',
            'Employment_Info_3', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3',
            'Insurance_History_1', 'Family_Hist_1', 'Medical_History_1', 'Medical_History_2',
            'Medical_History_4'
        ] + [f'Medical_Keyword_{i}' for i in range(1, 49)]
    
    def train_model(self, train_data):
        """Train the risk prediction model using Prudential dataset"""
        try:
            X_train = train_data[self.feature_columns]
            y_train = train_data['Response'] - 1  # Convert to 0-based indexing (0-7)
            
            # Handle missing values
            X_train = X_train.fillna(X_train.median())
            
            X_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_scaled, y_train)
            self.is_trained = True
            logger.info("Risk model trained successfully")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def calculate_risk_score(self, application_data):
        """Calculate risk score using ML model"""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        features = pd.DataFrame([application_data])[self.feature_columns]
        features = features.fillna(features.median())  # Handle any remaining missing values
        scaled_features = self.scaler.transform(features)
        risk_probs = self.model.predict_proba(scaled_features)[0]
        risk_score = np.sum(risk_probs * np.arange(8)) / 7  # Normalize to [0,1]
        return risk_score

class DecisionEngine:
    """Makes underwriting decisions based on risk scores and rules"""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        self.rules = {
            'high_risk_medical': lambda app: sum([app[f'Medical_Keyword_{i}'] for i in range(1, 49)]) > 5,
            'high_bmi': lambda app: app['BMI'] > 0.8,
            'low_income': lambda app: app['Employment_Info_1'] < 0.1
        }
    
    def make_decision(self, application_data, risk_score):
        """Make underwriting decision based on risk score and rules"""
        decision = {'status': '', 'reason': [], 'additional_info_needed': []}
        
        # Check for automatic rejection rules
        if self.rules['high_risk_medical'](application_data):
            decision['status'] = 'REJECTED'
            decision['reason'].append("Multiple high-risk medical conditions detected")
            return decision
            
        if self.rules['low_income'](application_data):
            decision['status'] = 'REJECTED'
            decision['reason'].append("Income below minimum threshold")
            return decision
            
        # Risk-based decision
        if risk_score > self.risk_thresholds['high']:
            decision['status'] = 'REJECTED'
            decision['reason'].append("High risk score")
        elif risk_score > self.risk_thresholds['medium']:
            decision['status'] = 'PENDING'
            decision['additional_info_needed'].append("Additional medical documentation")
            decision['additional_info_needed'].append("Detailed employment history")
        elif risk_score > self.risk_thresholds['low']:
            decision['status'] = 'APPROVED'
            decision['reason'].append("Acceptable risk profile")
        else:
            decision['status'] = 'APPROVED'
            decision['reason'].append("Low risk profile")
            
        # Additional checks
        if self.rules['high_bmi'](application_data):
            decision['additional_info_needed'].append("Medical examination required")
            if decision['status'] == 'APPROVED':
                decision['status'] = 'PENDING'
                
        return decision
    """Makes underwriting decisions based on risk scores and rules"""
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        self.rules = {
            'high_risk_medical': lambda app: sum([app[f'Medical_Keyword_{i}'] for i in range(1, 49)]) > 5,
            'high_bmi': lambda app: app['BMI'] > 0.8,
            'low_income': lambda app: app['Employment_Info_1'] < 0.1
        }
    
    def make_decision(self, application_data, risk_score):
        """Make underwriting decision based on risk score and rules"""
        decision = {'status': '', 'reason': [], 'additional_info_needed': []}
        
        # Check for automatic rejection rules
        if self.rules['high_risk_medical'](application_data):
            decision['status'] = 'REJECTED'
            decision['reason'].append("Multiple high-risk medical conditions detected")
            return decision
            
        if self.rules['low_income'](application_data):
            decision['status'] = 'REJECTED'
            decision['reason'].append("Income below minimum threshold")
            return decision
            
        # Risk-based decision
        if risk_score > self.risk_thresholds['high']:
            decision['status'] = 'REJECTED'
            decision['reason'].append("High risk score")
        elif risk_score > self.risk_thresholds['medium']:
            decision['status'] = 'PENDING'
            decision['additional_info_needed'].append("Additional medical documentation")
            decision['additional_info_needed'].append("Detailed employment history")
        elif risk_score > self.risk_thresholds['low']:
            decision['status'] = 'APPROVED'
            decision['reason'].append("Acceptable risk profile")
        else:
            decision['status'] = 'APPROVED'
            decision['reason'].append("Low risk profile")
            
        # Additional checks
        if self.rules['high_bmi'](application_data):
            decision['additional_info_needed'].append("Medical examination required")
            if decision['status'] == 'APPROVED':
                decision['status'] = 'PENDING'
                
        return decision
class UnderwritingSystem:
    """Main underwriting system orchestrating all components"""
    
    def __init__(self, train_data_path):
        self.processor = ApplicationProcessor()
        self.risk_evaluator = RiskEvaluator()
        self.decision_engine = DecisionEngine()
        
        # Initialize with training data from file path
        self._initialize_model(train_data_path)
    
    def _initialize_model(self, train_data_path):
        """Initialize ML model with Prudential training data from file path"""
        try:
            train_data = pd.read_csv(train_data_path)
            self.risk_evaluator.train_model(train_data)
        except Exception as e:
            logger.error(f"Error loading training data from {train_data_path}: {str(e)}")
            raise
    
    def process_application(self, application_data):
        """Process a single insurance application"""
        try:
            # Validate application
            is_valid, message = self.processor.validate_application(application_data)
            if not is_valid:
                return {
                    'status': 'ERROR',
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
                
            # Preprocess data
            processed_data = self.processor.preprocess_data(application_data)
            
            # Calculate risk score
            risk_score = self.risk_evaluator.calculate_risk_score(processed_data)
            
            # Make decision
            decision = self.decision_engine.make_decision(processed_data, risk_score)
            
            # Generate output
            output = {
                'status': decision['status'],
                'risk_score': float(risk_score),
                'reasons': decision['reason'],
                'additional_info_needed': decision['additional_info_needed'],
                'timestamp': datetime.now().isoformat(),
                'application_id': application_data.get('Id', 'N/A')
            }
            
            logger.info(f"Application {output['application_id']} processed: {output['status']}")
            return output
            
        except Exception as e:
            logger.error(f"Error processing application: {str(e)}")
            return {
                'status': 'ERROR',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def test_model(self, test_data_path):
        """Test the model using test.csv from file path"""
        try:
            test_data = pd.read_csv(test_data_path)
            results = []
            for _, row in test_data.iterrows():
                result = self.process_application(row.to_dict())
                results.append(result)
            logger.info(f"Processed {len(results)} test applications")
            return results
        except Exception as e:
            logger.error(f"Error testing model with {test_data_path}: {str(e)}")
            return [{
                'status': 'ERROR',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }]
def process_application_interface(application_id, ins_age, ht, wt, bmi, employment_info_1, 
                               employment_info_2, employment_info_3, insured_info_1, insured_info_2, 
                               insured_info_3, insurance_history_1, family_hist_1, medical_history_1, 
                               medical_history_2, medical_history_4, medical_keyword_1, medical_keyword_2, 
                               medical_keyword_3, train_data_path):
    """Process application through Gradio interface"""
    underwriting_system = UnderwritingSystem(train_data_path)
    
    # Create application dictionary
    application_data = {
        'Id': application_id,
        'Ins_Age': ins_age,
        'Ht': ht,
        'Wt': wt,
        'BMI': bmi,
        'Employment_Info_1': employment_info_1,
        'Employment_Info_2': employment_info_2,
        'Employment_Info_3': employment_info_3,
        'InsuredInfo_1': insured_info_1,
        'InsuredInfo_2': insured_info_2,
        'InsuredInfo_3': insured_info_3,
        'Insurance_History_1': insurance_history_1,
        'Family_Hist_1': family_hist_1,
        'Medical_History_1': medical_history_1,
        'Medical_History_2': medical_history_2,
        'Medical_History_4': medical_history_4,
        'Medical_Keyword_1': medical_keyword_1,
        'Medical_Keyword_2': medical_keyword_2,
        'Medical_Keyword_3': medical_keyword_3,
        **{f'Medical_Keyword_{i}': 0 for i in range(4, 49)}  # Default remaining keywords to 0
    }
    
    result = underwriting_system.process_application(application_data)
    return json.dumps(result, indent=2)
def main():
    """Main function to launch Gradio interface and demonstrate testing"""
    train_data_path = "/content/train.csv"  # Adjust path as needed
    
    # Launch Gradio interface
    iface = gr.Interface(
        fn=process_application_interface,
        inputs=[
          gr.Textbox(label="Application ID", value="APP001"),
          gr.Slider(minimum=0, maximum=1, step=0.01, label="Insured Age (Normalized)", value=0.5),
          gr.Slider(minimum=0, maximum=1, step=0.01, label="Height (Normalized)", value=0.5),
          gr.Slider(minimum=0, maximum=1, step=0.01, label="Weight (Normalized)", value=0.5),
          gr.Slider(minimum=0, maximum=1, step=0.01, label="BMI (Normalized)", value=0.5),
          gr.Slider(minimum=0, maximum=1, step=0.01, label="Employment Info 1 (Income, Normalized)", value=0.5),
          gr.Dropdown(choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], label="Employment Info 2", value=1),
          gr.Dropdown(choices=[1, 2, 3], label="Employment Info 3", value=1),
          gr.Dropdown(choices=[1, 2, 3], label="Insured Info 1", value=1),
          gr.Dropdown(choices=[1, 2, 3], label="Insured Info 2", value=1),
          gr.Dropdown(choices=[1, 2, 3, 4, 5, 6, 7, 8], label="Insured Info 3", value=1),
          gr.Dropdown(choices=[1, 2], label="Insurance History 1", value=1),
          gr.Dropdown(choices=[1, 2, 3], label="Family History 1", value=1),
          gr.Number(label="Medical History 1", value=1),
          gr.Number(label="Medical History 2", value=1),
          gr.Dropdown(choices=[1, 2], label="Medical History 4", value=1),
          gr.Checkbox(label="Medical Keyword 1 (Condition Present)", value=False),
          gr.Checkbox(label="Medical Keyword 2 (Condition Present)", value=False),
          gr.Checkbox(label="Medical Keyword 3 (Condition Present)", value=False),
          gr.Textbox(label="Train Data Path", value="/content/train.csv")
      ],
        outputs=gr.JSON(label="Underwriting Decision"),
        title="Life Insurance Underwriting System",
        description="Enter application details to get an underwriting decision. Ensure train.csv is available at the specified path."
    )
    
    underwriting_system = UnderwritingSystem(train_data_path)
    
    # Launch Gradio interface
    iface.launch()

if __name__ == "__main__":
    main()