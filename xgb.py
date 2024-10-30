import pandas as pd
import xgboost as xgb
import streamlit as st

# 读取训练集数据
train_data = pd.read_csv('train_data.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Sex', 'Primary Site', 'Histologic Type',
                'T stage', 'Surgery', 'Chemotherapy',
                'Bone metastasis', 'Brain metastasis', 'Lung metastasis']]
y = train_data['Vital status']

# 创建并训练XGBoost模型
xgb_params = {
    'alpha': 0.37747316967433386,
    'eta': 0.583195161883046,
    'gamma': 2.2996141540103423,
    'max_delta_step': 5,
    'max_depth': 4,
    'min_child_weight': 3,
    'n_estimators': 66,
    'subsample': 0.89985728184,
}

xgb_model = xgb.XGBClassifier(**xgb_params)

# 特征映射
class_mapping = {0: "Alive", 1: "Dead"}
age_mapper = {'57-78': 1, '>78': 2, '<57': 3}
sex_mapper = {'male': 1, 'female': 2}
primary_site_mapper = {"Upper third of esophagus": 1, "Middle third of esophagus": 2, "Lower third of esophagus": 3, "Overlapping lesion of esophagus": 4}
histologic_type_mapper = {"Adenocarcinoma": 1, "Squamous-cell carcinoma": 2}
t_stage_mapper = {"T1": 4, "T2": 1, "T3": 2, "T4": 3}
surgery_mapper = {"NO": 2, "Yes": 1}
chemotherapy_mapper = {"NO": 2, "Yes": 1}
bone_metastasis_mapper = {"NO": 2, "Yes": 1}
brain_metastasis_mapper = {"NO": 2, "Yes": 1}
lung_metastasis_mapper = {"NO": 2, "Yes": 1}

# 训练XGBoost模型
xgb_model.fit(X, y)

# 预测函数
def predict_Vital_status(age, sex, primary_site, histologic_type,
                         t_stage, surgery, chemotherapy,
                         bone_metastasis, brain_metastasis, lung_metastasis):
    input_data = pd.DataFrame({
        'Age': [age_mapper[age]],
        'Sex': [sex_mapper[sex]],
        'Primary Site': [primary_site_mapper[primary_site]],
        'Histologic Type': [histologic_type_mapper[histologic_type]],
        'T stage': [t_stage_mapper[t_stage]],
        'Surgery': [surgery_mapper[surgery]],
        'Chemotherapy': [chemotherapy_mapper[chemotherapy]],
        'Bone metastasis': [bone_metastasis_mapper[bone_metastasis]],
        'Brain metastasis': [brain_metastasis_mapper[brain_metastasis]],
        'Lung metastasis': [lung_metastasis_mapper[lung_metastasis]]
    })
    prediction = xgb_model.predict(input_data)[0]
    probability = xgb_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("6-month survival of ECLM patients based on XGBoost")
st.sidebar.write("Variables")

age = st.sidebar.selectbox("Age", options=list(age_mapper.keys()))  # 使用选择框
sex = st.sidebar.selectbox("Sex", options=list(sex_mapper.keys()))
primary_site = st.sidebar.selectbox("Primary Site", options=list(primary_site_mapper.keys()))
histologic_type = st.sidebar.selectbox("Histologic Type", options=list(histologic_type_mapper.keys()))
t_stage = st.sidebar.selectbox("T stage", options=list(t_stage_mapper.keys()))
surgery = st.sidebar.selectbox("Surgery", options=list(surgery_mapper.keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(chemotherapy_mapper.keys()))
bone_metastasis = st.sidebar.selectbox("Bone metastasis", options=list(bone_metastasis_mapper.keys()))
brain_metastasis = st.sidebar.selectbox("Brain metastasis", options=list(brain_metastasis_mapper.keys()))
lung_metastasis = st.sidebar.selectbox("Lung metastasis", options=list(lung_metastasis_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_Vital_status(
        age, sex, primary_site, histologic_type, t_stage, surgery, chemotherapy,
        bone_metastasis, brain_metastasis, lung_metastasis
    )

    st.write("Predicted Vital Status:", prediction)
    st.write("Probability of 6-month survival is:", probability)
