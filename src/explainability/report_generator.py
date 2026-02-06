"""
Human-readable report generator for loan decisions.

Generates natural language explanations in Korean and English.
"""
from typing import Dict, Any, Optional
from datetime import datetime


class LoanDecisionReportGenerator:
    """
    Generates human-readable loan decision reports.
    
    Creates bilingual (Korean/English) reports explaining:
    1. The decision and confidence level
    2. Most similar customer prototypes
    3. Key factors contributing to the decision
    4. Comparison with input features
    """
    
    # Feature display names
    FEATURE_NAMES = {
        'person_age': ('나이', 'Age'),
        'person_income': ('연소득', 'Annual Income'),
        'person_emp_length': ('근속연수', 'Employment Length'),
        'loan_amnt': ('대출금액', 'Loan Amount'),
        'loan_int_rate': ('이자율', 'Interest Rate'),
        'loan_percent_income': ('소득대비대출비율', 'Loan-to-Income Ratio'),
        'cb_person_cred_hist_length': ('신용이력기간', 'Credit History Length'),
        'person_home_ownership': ('주거형태', 'Home Ownership'),
        'loan_intent': ('대출목적', 'Loan Intent'),
        'loan_grade': ('신용등급', 'Loan Grade'),
        'cb_person_default_on_file': ('과거연체이력', 'Past Default')
    }
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.5,
        'high': 0.7
    }
    
    def __init__(self, explainer=None):
        """
        Args:
            explainer: PrototypeExplainer instance (optional)
        """
        self.explainer = explainer
    
    def generate_report(
        self,
        explanation: Dict[str, Any],
        applicant_id: Optional[str] = None,
        language: str = 'both'  # 'ko', 'en', or 'both'
    ) -> str:
        """
        Generate a formatted loan decision report.
        
        Args:
            explanation: Output from PrototypeExplainer.explain_single()
            applicant_id: Optional applicant identifier
            language: Report language ('ko', 'en', or 'both')
            
        Returns:
            Formatted report string
        """
        prediction = explanation['prediction']
        input_features = explanation['input_features']
        top_prototypes = explanation['top_prototypes']
        
        # Determine decision and risk level
        if prediction < self.RISK_THRESHOLDS['low']:
            decision_ko, decision_en = '승인 권고', 'Recommended for Approval'
            risk_ko, risk_en = '낮음', 'Low'
        elif prediction < self.RISK_THRESHOLDS['medium']:
            decision_ko, decision_en = '조건부 승인 검토', 'Conditional Review'
            risk_ko, risk_en = '보통', 'Medium'
        elif prediction < self.RISK_THRESHOLDS['high']:
            decision_ko, decision_en = '신중한 검토 필요', 'Requires Careful Review'
            risk_ko, risk_en = '높음', 'High'
        else:
            decision_ko, decision_en = '거절 권고', 'Recommended for Rejection'
            risk_ko, risk_en = '매우 높음', 'Very High'
        
        report_parts = []
        
        # Korean report
        if language in ['ko', 'both']:
            report_parts.append(self._generate_korean_report(
                prediction, decision_ko, risk_ko,
                input_features, top_prototypes, applicant_id
            ))
        
        # English report
        if language in ['en', 'both']:
            report_parts.append(self._generate_english_report(
                prediction, decision_en, risk_en,
                input_features, top_prototypes, applicant_id
            ))
        
        return '\n\n'.join(report_parts)
    
    def _generate_korean_report(
        self,
        prediction: float,
        decision: str,
        risk_level: str,
        input_features: Dict[str, Any],
        top_prototypes: list,
        applicant_id: Optional[str]
    ) -> str:
        """Generate Korean language report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    대출 심사 결과 보고서 (AI 분석)                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 생성일시: {timestamp}                                            ║
║ 신청번호: {applicant_id or 'N/A':<15}                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

■ 심사 결과
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  결정: {decision:<20}                                          │
  │  부도 위험 확률: {prediction*100:>5.1f}%                                          │
  │  위험 수준: {risk_level:<10}                                               │
  └─────────────────────────────────────────────────────────────────────────┘

■ 신청자 정보
  ┌─────────────────────────────────────────────────────────────────────────┐"""
        
        for key, value in input_features.items():
            name_ko = self.FEATURE_NAMES.get(key, (key, key))[0]
            if isinstance(value, float):
                if 'income' in key:
                    value_str = f"${value:,.0f}"
                elif 'rate' in key or 'percent' in key:
                    value_str = f"{value:.2f}%"
                else:
                    value_str = f"{value:.1f}"
            else:
                value_str = str(value)
            report += f"\n  │  {name_ko:<20}: {value_str:<30}│"
        
        report += """
  └─────────────────────────────────────────────────────────────────────────┘

■ 유사 고객 유형 분석 (프로토타입 기반)"""
        
        for i, proto in enumerate(top_prototypes, 1):
            name = proto.get('name_ko', f'프로토타입 {proto["index"]}')
            desc = proto.get('description_ko', '')
            sim = proto['similarity'] * 100
            contrib = proto['contribution']
            
            report += f"""
  
  {i}. {name}
     - 유사도: {sim:.1f}%
     - 기여도: {contrib:.4f}"""
            
            if desc:
                report += f"""
     - 설명: {desc}"""
        
        report += """

■ 결정 근거
  본 분석은 AI 모델의 프로토타입 기반 추론 결과입니다.
  신청자의 특성이 위 유형들과 유사할수록 해당 유형의 부도 패턴을 따를 가능성이 있습니다.
  최종 결정은 담당자의 종합적인 판단을 통해 이루어져야 합니다.

══════════════════════════════════════════════════════════════════════════════════
"""
        return report
    
    def _generate_english_report(
        self,
        prediction: float,
        decision: str,
        risk_level: str,
        input_features: Dict[str, Any],
        top_prototypes: list,
        applicant_id: Optional[str]
    ) -> str:
        """Generate English language report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LOAN DECISION REPORT (AI Analysis)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Generated: {timestamp}                                           ║
║ Application ID: {applicant_id or 'N/A':<15}                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

■ Decision Summary
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Decision: {decision:<35}                               │
  │  Default Probability: {prediction*100:>5.1f}%                                     │
  │  Risk Level: {risk_level:<15}                                          │
  └─────────────────────────────────────────────────────────────────────────┘

■ Applicant Information
  ┌─────────────────────────────────────────────────────────────────────────┐"""
        
        for key, value in input_features.items():
            name_en = self.FEATURE_NAMES.get(key, (key, key))[1]
            if isinstance(value, float):
                if 'income' in key:
                    value_str = f"${value:,.0f}"
                elif 'rate' in key or 'percent' in key:
                    value_str = f"{value:.2f}%"
                else:
                    value_str = f"{value:.1f}"
            else:
                value_str = str(value)
            report += f"\n  │  {name_en:<25}: {value_str:<25}│"
        
        report += """
  └─────────────────────────────────────────────────────────────────────────┘

■ Similar Customer Profile Analysis (Prototype-Based)"""
        
        for i, proto in enumerate(top_prototypes, 1):
            name = proto.get('name_en', f'Prototype {proto["index"]}')
            desc = proto.get('description_en', '')
            sim = proto['similarity'] * 100
            contrib = proto['contribution']
            
            report += f"""
  
  {i}. {name}
     - Similarity: {sim:.1f}%
     - Contribution: {contrib:.4f}"""
            
            if desc:
                report += f"""
     - Description: {desc}"""
        
        report += """

■ Reasoning
  This analysis is based on prototype-based reasoning from the AI model.
  The more similar the applicant's characteristics to the prototypes above,
  the more likely they are to follow similar default patterns.
  Final decisions should be made through comprehensive human review.

══════════════════════════════════════════════════════════════════════════════════
"""
        return report
    
    def generate_summary(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured summary for API/JSON responses.
        
        Args:
            explanation: Output from PrototypeExplainer.explain_single()
            
        Returns:
            Structured summary dict
        """
        prediction = explanation['prediction']
        
        # Determine risk category
        if prediction < 0.3:
            risk_category = 'low'
        elif prediction < 0.5:
            risk_category = 'medium_low'
        elif prediction < 0.7:
            risk_category = 'medium_high'
        else:
            risk_category = 'high'
        
        return {
            'prediction': prediction,
            'risk_category': risk_category,
            'confidence': 1 - abs(prediction - 0.5) * 2,  # Higher near 0 or 1
            'top_prototypes': [
                {
                    'index': p['index'],
                    'name': p.get('name_en', f"Prototype {p['index']}"),
                    'similarity': p['similarity'],
                    'contribution': p['contribution']
                }
                for p in explanation['top_prototypes']
            ],
            'key_factors': self._extract_key_factors(explanation)
        }
    
    def _extract_key_factors(self, explanation: Dict[str, Any]) -> list:
        """Extract key factors contributing to the decision."""
        factors = []
        input_features = explanation['input_features']
        
        # Check high-risk indicators
        if input_features.get('cb_person_default_on_file') == 'Y':
            factors.append({
                'factor': 'past_default',
                'impact': 'negative',
                'description': 'Past default history on file'
            })
        
        if input_features.get('loan_percent_income', 0) > 0.3:
            factors.append({
                'factor': 'high_dti',
                'impact': 'negative', 
                'description': f"High loan-to-income ratio: {input_features['loan_percent_income']:.1%}"
            })
        
        if input_features.get('person_emp_length', 0) < 2:
            factors.append({
                'factor': 'short_employment',
                'impact': 'negative',
                'description': 'Short employment history'
            })
        
        # Check positive indicators
        if input_features.get('cb_person_cred_hist_length', 0) > 10:
            factors.append({
                'factor': 'long_credit_history',
                'impact': 'positive',
                'description': 'Established credit history'
            })
        
        if input_features.get('loan_grade', 'G') in ['A', 'B']:
            factors.append({
                'factor': 'good_loan_grade',
                'impact': 'positive',
                'description': f"Favorable loan grade: {input_features['loan_grade']}"
            })
        
        return factors
