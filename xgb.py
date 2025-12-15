import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

# 读取训练集数据
train_data = pd.read_csv('train_data.csv')

# 定义全部14个特征列
feature_columns = [
    'Age', 'Race', 'Sex', 'Primary Site', 'Histologic Type', 'Grade',
    'T stage', 'N stage', 'Surgery', 'Radiation', 'Chemotherapy',
    'Bone metastasis', 'Brain metastasis', 'Lung metastasis'
]

# 分离输入特征和目标变量
X = train_data[feature_columns]
y = train_data['Vital status']

# 创建逻辑回归模型（使用指定的最佳参数）
lr_params = {
    'C': 0.001,
    'penalty': 'l2',
    'solver': 'liblinear',
    'class_weight': 'balanced',
    'random_state': 42
}

lr_model = LogisticRegression(**lr_params)

# 特征映射字典
class_mapping = {0: "Alive", 1: "Dead"}

age_mapper = {'57-78': 1, '>78': 2, '<57': 3}
race_mapper = {'White': 1, 'Black': 2, 'Others': 3}
sex_mapper = {'male': 1, 'female': 2}
primary_site_mapper = {
    "Upper third of esophagus": 1,
    "Middle third of esophagus": 2,
    "Lower third of esophagus": 3,
    "Overlapping lesion of esophagus": 4
}
histologic_type_mapper = {"Adenocarcinoma": 1, "Squamous-cell carcinoma": 2}
grade_mapper = {
    "Grade II": 1,
    "Grade III": 2,
    "Grade IV": 3,
    "Grade I": 4
}
t_stage_mapper = {"T1": 4, "T2": 1, "T3": 2, "T4": 3}
n_stage_mapper = {"N1": 1, "N2": 2, "N3": 3, "N0": 4}
surgery_mapper = {"NO": 2, "Yes": 1}
radiation_mapper = {"Yes": 1, "No": 0}
chemotherapy_mapper = {"NO": 2, "Yes": 1}
bone_metastasis_mapper = {"NO": 2, "Yes": 1}
brain_metastasis_mapper = {"NO": 2, "Yes": 1}
lung_metastasis_mapper = {"NO": 2, "Yes": 1}

# 对训练集 X 进行编码
X_encoded = X.copy()
X_encoded['Age'] = X_encoded['Age'].map(age_mapper)
X_encoded['Race'] = X_encoded['Race'].map(race_mapper)
X_encoded['Sex'] = X_encoded['Sex'].map(sex_mapper)
X_encoded['Primary Site'] = X_encoded['Primary Site'].map(primary_site_mapper)
X_encoded['Histologic Type'] = X_encoded['Histologic Type'].map(histologic_type_mapper)
X_encoded['Grade'] = X_encoded['Grade'].map(grade_mapper)  
X_encoded['T stage'] = X_encoded['T stage'].map(t_stage_mapper)
X_encoded['N stage'] = X_encoded['N stage'].map(n_stage_mapper)
X_encoded['Surgery'] = X_encoded['Surgery'].map(surgery_mapper)
X_encoded['Radiation'] = X_encoded['Radiation'].map(radiation_mapper)
X_encoded['Chemotherapy'] = X_encoded['Chemotherapy'].map(chemotherapy_mapper)
X_encoded['Bone metastasis'] = X_encoded['Bone metastasis'].map(bone_metastasis_mapper)
X_encoded['Brain metastasis'] = X_encoded['Brain metastasis'].map(brain_metastasis_mapper)
X_encoded['Lung metastasis'] = X_encoded['Lung metastasis'].map(lung_metastasis_mapper)


# 训练模型
lr_model.fit(X_encoded, y)

# 预测函数
def predict_Vital_status(age, race, sex, primary_site, histologic_type, grade,
                         t_stage, n_stage, surgery, radiation, chemotherapy,
                         bone_metastasis, brain_metastasis, lung_metastasis):
    input_data = pd.DataFrame({
        'Age': [age_mapper[age]],
        'Race': [race_mapper[race]],
        'Sex': [sex_mapper[sex]],
        'Primary Site': [primary_site_mapper[primary_site]],
        'Histologic Type': [histologic_type_mapper[histologic_type]],
        'Grade': [grade_mapper[grade]], 
        'T stage': [t_stage_mapper[t_stage]],
        'N stage': [n_stage_mapper[n_stage]],
        'Surgery': [surgery_mapper[surgery]],
        'Radiation': [radiation_mapper[radiation]],
        'Chemotherapy': [chemotherapy_mapper[chemotherapy]],
        'Bone metastasis': [bone_metastasis_mapper[bone_metastasis]],
        'Brain metastasis': [brain_metastasis_mapper[brain_metastasis]],
        'Lung metastasis': [lung_metastasis_mapper[lung_metastasis]]
    })
    prediction = lr_model.predict(input_data)[0]
    prob_dead = lr_model.predict_proba(input_data)[0][1]  # P(Dead)
    class_label = class_mapping[prediction]
    return class_label, prob_dead

# Streamlit 界面
st.title("6-month survival of ECLM patients based on Logistic Regression")
st.sidebar.write("Variables")

age = st.sidebar.selectbox("Age", options=list(age_mapper.keys()))
race = st.sidebar.selectbox("Race", options=list(race_mapper.keys()))
sex = st.sidebar.selectbox("Sex", options=list(sex_mapper.keys()))
primary_site = st.sidebar.selectbox("Primary Site", options=list(primary_site_mapper.keys()))
histologic_type = st.sidebar.selectbox("Histologic Type", options=list(histologic_type_mapper.keys()))
grade = st.sidebar.selectbox("Grade", options=list(grade_mapper.keys()))  
t_stage = st.sidebar.selectbox("T stage", options=list(t_stage_mapper.keys()))
n_stage = st.sidebar.selectbox("N stage", options=list(n_stage_mapper.keys()))
surgery = st.sidebar.selectbox("Surgery", options=list(surgery_mapper.keys()))
radiation = st.sidebar.selectbox("Radiation", options=list(radiation_mapper.keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(chemotherapy_mapper.keys()))
bone_metastasis = st.sidebar.selectbox("Bone metastasis", options=list(bone_metastasis_mapper.keys()))
brain_metastasis = st.sidebar.selectbox("Brain metastasis", options=list(brain_metastasis_mapper.keys()))
lung_metastasis = st.sidebar.selectbox("Lung metastasis", options=list(lung_metastasis_mapper.keys()))

if st.button("Predict"):
    prediction, prob_dead = predict_Vital_status(
        age, race, sex, primary_site, histologic_type, grade,
        t_stage, n_stage, surgery, radiation, chemotherapy,
        bone_metastasis, brain_metastasis, lung_metastasis
    )
    prob_survival = 1 - prob_dead

    st.write("Predicted Vital Status:", prediction)
    st.write(f"Probability of 6-month survival: {prob_survival:.4f}")
    st.write(f"Probability of death within 6 months: {prob_dead:.4f}")
