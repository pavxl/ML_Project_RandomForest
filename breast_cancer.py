import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
dataset = load_breast_cancer()
data = pd.DataFrame(dataset['data'], columns=columns)
data['cancer'] = dataset['target']
display(data.head())
# display(data.info())
# display(data.isna().sum())
# display(data.describe())

"""Разделение набора данных"""
X = data.drop('cancer', axis=1)
y = data['cancer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state = 2023, stratify=y)

"""Масштабирование"""

ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
y_train = np.array(y_train)

"""Обучение базовой модели"""
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled, y_train)
display(rfc.score(X_train_scaled, y_train))

feats = {feature: importance for feature, importance in zip(data.columns, rfc.feature_importances_)}

importances = pd.DataFrame.from_dict(feats, orient='index', columns=['Gini-Importance'])
importances = importances.sort_values(by='Gini-Importance', ascending=False).reset_index()
importances = importances.rename(columns={'index': 'Features'})

sns.set(font_scale=1.7)
fig, ax = plt.subplots(figsize=(30, 15))
sns.barplot(x='Gini-Importance', y='Features', data=importances, color='skyblue')
ax.set_xlabel('Importance', fontsize=25, weight='bold')
ax.set_ylabel('Features', fontsize=25, weight='bold')
ax.set_title('Feature Importance', fontsize=25, weight='bold')

plt.show()

# display(importances)

"""Улучшение модели методом главных компонент"""
pca_test = PCA(n_components=30)
X_train_scaled_pca = pca_test.fit_transform(X_train_scaled)
X_test_scaled_pca = pca_test.transform(X_test_scaled)

sns.set(style='whitegrid')
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.axvline(linewidth=4, color='r', linestyle='--', x=10, ymin=0, ymax=1)
plt.show()

evr = pca_test.explained_variance_ratio_
cvr = np.cumsum(evr)

pca_df = pd.DataFrame({'Cumulative Variance Ratio': cvr, 'Explained Variance Ratio': evr})
display(pca_df.head(10))

pca = PCA(n_components=10)
X_train_scaled_pca = pca.fit_transform(X_train_scaled)
X_test_scaled_pca = pca.transform(X_test_scaled)

"""Представляем компоненты как линейную комбинацию исходных переменных с весами"""
pca_dims = ['PCA Component {}'.format(x) for x in range(len(pca_df))]
pca_test_df = pd.DataFrame(pca_test.components_.T[:, :10], columns=pca_dims)
pca_test_df.head(10)


"""Обучение RandomForest-модели"""
rfc_2 = RandomForestClassifier()
rfc_2.fit(X_train_scaled_pca, y_train)
display(rfc_2.score(X_train_scaled_pca, y_train))

"""Оптимизация гиперпараметров RandSearchCV"""
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(start=1, stop=15, num=15)]
min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
bootstrap = [True, False]

param_dist = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

rs = RandomizedSearchCV(
    rfc_2,
    param_dist,
    n_iter=100,
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=0
)

rs.fit(X_train_scaled_pca, y_train)
rs.best_params_


rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
rs_df = rs_df.drop([
            'mean_fit_time',
            'std_fit_time',
            'mean_score_time',
            'std_score_time',
            'params',
            'split0_test_score',
            'split1_test_score',
            'split2_test_score',
            'std_test_score'],
            axis=1)
rs_df.head(10)

"""Выявление лучших гиперпараметров"""
fig, axs = plt.subplots(ncols=3, nrows=2)
sns.set(style="whitegrid", color_codes=True, font_scale=2)
fig.set_size_inches(30, 25)

sns.barplot(x='param_n_estimators', y='mean_test_score', data=rs_df, ax=axs[0, 0], color='lightgrey')
axs[0, 0].set_ylim([0.83, 0.93])
axs[0, 0].set_title('n_estimators', size=30, weight='bold')

sns.barplot(x='param_min_samples_split', y='mean_test_score', data=rs_df, ax=axs[0, 1], color='coral')
axs[0, 1].set_ylim([0.85, 0.93])
axs[0, 1].set_title('min_samples_split', size=30, weight='bold')

sns.barplot(x='param_min_samples_leaf', y='mean_test_score', data=rs_df, ax=axs[0, 2], color='lightgreen')
axs[0, 2].set_ylim([0.80, 0.93])
axs[0, 2].set_title('min_samples_leaf', size=30, weight='bold')

sns.barplot(x='param_max_features', y='mean_test_score', data=rs_df, ax=axs[1, 0], color='wheat')
axs[1, 0].set_ylim([0.88, 0.92])
axs[1, 0].set_title('max_features', size=30, weight='bold')

sns.barplot(x='param_max_depth', y='mean_test_score', data=rs_df, ax=axs[1, 1], color='lightpink')
axs[1, 1].set_ylim([0.80, 0.93])
axs[1, 1].set_title('max_depth', size=30, weight='bold')

sns.barplot(x='param_bootstrap', y='mean_test_score', data=rs_df, ax=axs[1, 2], color='skyblue')
axs[1, 2].set_ylim([0.88, 0.92])
axs[1, 2].set_title('bootstrap', size=30, weight='bold')

plt.tight_layout()
plt.show()

"""Оптимизация гиперпараметров, выбор лучших"""
n_estimators = [300, 500, 700]
max_features = ['sqrt']
max_depth = [2, 3, 7, 11, 15]
min_samples_split = [2, 3, 4, 22, 23, 24]
min_samples_leaf = [2, 3, 4, 5, 6, 7]
bootstrap = [False]

param_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

gs = GridSearchCV(rfc_2, param_grid, cv=3, verbose=1, n_jobs=-1)
gs.fit(X_train_scaled_pca, y_train)
rfc_3 = gs.best_estimator_
gs.best_params_


y_pred = rfc.predict(X_test_scaled)
y_pred_pca = rfc_2.predict(X_test_scaled_pca)
y_pred_gs = gs.best_estimator_.predict(X_test_scaled_pca)

conf_matrix_baseline = pd.DataFrame(confusion_matrix(y_test, y_pred), index=['actual 0', 'actual 1'], columns=['predicted 0', 'predicted 1'])
conf_matrix_baseline_pca = pd.DataFrame(confusion_matrix(y_test, y_pred_pca), index=['actual 0', 'actual 1'], columns=['predicted 0', 'predicted 1'])
conf_matrix_tuned_pca = pd.DataFrame(confusion_matrix(y_test, y_pred_gs), index=['actual 0', 'actual 1'], columns=['predicted 0', 'predicted 1'])

display(conf_matrix_baseline)
display('Baseline Random Forest recall score:', recall_score(y_test, y_pred))

display(conf_matrix_baseline_pca)
display('Baseline Random Forest With PCA recall score:', recall_score(y_test, y_pred_pca))

display(conf_matrix_tuned_pca)
display('Hyperparameter Tuned Random Forest With PCA Reduced Dimensionality recall score:', recall_score(y_test, y_pred_gs))
