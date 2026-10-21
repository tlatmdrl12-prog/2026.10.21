"""
귀화 가능성 예측 대시보드 - 데이터 생성 코드
================================================
대시보드에 나오는 모든 숫자를 원본 CSV에서 다시 계산합니다.

입력 : 귀화예측_상위10컬럼_ID가중치포함.csv  (이 파일과 같은 폴더에 두기)
출력 : dashboard_data/ 폴더에 CSV 7개 + tree_model.json
       화면에 모델 성능표와 예측 예시 출력

실행 : python naturalization_dashboard_data.py
필요 : pandas, numpy, scikit-learn
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "귀화예측_상위10컬럼_ID가중치포함.csv"
OUT_DIR = "dashboard_data"
SURVEY_YEAR = 2021
코드컬럼 = ["혼인상태", "성별", "출생시국적코드(외국)", "장애인복지카드유무", "최초입국여부"]

os.makedirs(OUT_DIR, exist_ok=True)


# =================================================================
# 1. 데이터 준비 (노트북과 같은 순서)
# =================================================================
raw = pd.read_csv(DATA_PATH)                     # 원본 (귀화율 표에 사용)
중앙값 = raw.median(numeric_only=True)              # 빈칸을 채울 값 (예측기에도 사용)

df = raw.fillna(중앙값)
df["체류기간"] = SURVEY_YEAR - df["최초입국연도_년"]
df = df.drop(columns=["최초입국연도_년"])
df = pd.get_dummies(df, columns=코드컬럼, drop_first=True)

X = df.drop(columns=["국적취득여부", "최종학력(학년이 아님)", "이민귀화자와관계", "가구ID", "가중치"])
y = df["국적취득여부"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)        # 로지스틱·KNN용 (표준화)
X_test_s = scaler.transform(X_test)              # 결정트리는 표준화가 필요 없음

print(f"분석 대상 {len(raw):,}명 / 훈련 {len(X_train):,}명 / 시험 {len(X_test):,}명 / 피처 {X.shape[1]}개")


# =================================================================
# 2. 모델 6개 학습 + 평가
#    핵심: 학습할 때와 평가할 때 같은 형태(표준화 여부)의 데이터를 넣는다
# =================================================================
결과 = []

def 평가(이름, 설정, 모델, 훈련X, 시험X, 비고):
    확률 = 모델.predict_proba(시험X)[:, 1]
    훈련정확도 = 모델.score(훈련X, y_train)
    시험정확도 = 모델.score(시험X, y_test)
    결과.append({
        "model": 이름, "setting": 설정,
        "train_acc": round(훈련정확도, 4),
        "test_acc": round(시험정확도, 4),
        "test_auc": round(roc_auc_score(y_test, 확률), 4),
        "gap": round(훈련정확도 - 시험정확도, 4),
        "note": 비고,
    })
    return 모델

# (1) 로지스틱 회귀
평가("로지스틱 회귀", "max_iter=2000, 표준화",
     LogisticRegression(max_iter=2000).fit(X_train_s, y_train),
     X_train_s, X_test_s, "노트북 16번 셀을 표준화 데이터로 학습하도록 고쳐 계산")

# (2) 결정트리 기본
평가("결정트리", "max_depth=5",
     DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train),
     X_train, X_test, "노트북 23·24번 셀 혼동행렬과 같은 결과")

# (3) 결정트리 GridSearchCV 최적값
평가("결정트리 GridSearchCV", "max_depth=11, min_impurity_decrease=0.0005, min_samples_leaf=1",
     DecisionTreeClassifier(max_depth=11, min_impurity_decrease=0.0005,
                            min_samples_leaf=1, random_state=42).fit(X_train, y_train),
     X_train, X_test, "노트북 45·47번 셀과 같은 값")

# (4) 결정트리 RandomizedSearchCV 최적값  ← 대시보드 예측기에 쓰는 모델
best = DecisionTreeClassifier(max_depth=14, min_impurity_decrease=0.0003517822958253642,
                              min_samples_leaf=2, min_samples_split=4, random_state=42)
best.fit(X_train, y_train)
평가("결정트리 RandomizedSearchCV",
     "max_depth=14, min_impurity_decrease≈0.00035, min_samples_leaf=2, min_samples_split=4",
     best, X_train, X_test, "예측기에 쓰는 모델")

# (5) KNN 기본
평가("KNN", "n_neighbors=5, 표준화",
     KNeighborsClassifier().fit(X_train_s, y_train),
     X_train_s, X_test_s, "노트북 58번 셀은 원래 값으로 학습해 점수가 다름")

# (6) KNN GridSearchCV
grid = GridSearchCV(
    make_pipeline(StandardScaler(), KNeighborsClassifier()),
    {"kneighborsclassifier__n_neighbors": [5, 15, 31, 51, 101],
     "kneighborsclassifier__weights": ["uniform", "distance"]},
    cv=5, scoring="roc_auc")
grid.fit(X_train_s, y_train)
k = grid.best_params_["kneighborsclassifier__n_neighbors"]
w = grid.best_params_["kneighborsclassifier__weights"]
평가("KNN GridSearchCV", f"n_neighbors={k}, weights={w}",
     grid.best_estimator_, X_train_s, X_test_s, "노트북 59번 셀 설정")

model_metrics = pd.DataFrame(결과)
model_metrics.to_csv(f"{OUT_DIR}/model_metrics.csv", index=False, encoding="utf-8-sig")
print("\n[모델별 정확도]")
print(model_metrics[["model", "train_acc", "test_acc", "test_auc", "gap"]].to_string(index=False))


# =================================================================
# 3. 최적 모델의 혼동행렬 (시험 데이터)
# =================================================================
cm = confusion_matrix(y_test, best.predict(X_test))
confusion = pd.DataFrame([
    {"actual": "미귀화", "predicted": "미귀화", "n": cm[0, 0]},
    {"actual": "미귀화", "predicted": "귀화",   "n": cm[0, 1]},
    {"actual": "귀화",   "predicted": "미귀화", "n": cm[1, 0]},
    {"actual": "귀화",   "predicted": "귀화",   "n": cm[1, 1]},
])
confusion.to_csv(f"{OUT_DIR}/confusion.csv", index=False, encoding="utf-8-sig")
print("\n[혼동행렬] 행=실제(미귀화, 귀화), 열=예측(미귀화, 귀화)")
print(cm)


# =================================================================
# 4. 특성 중요도 (0/1로 나눈 코드 컬럼은 원래 컬럼별로 합치기)
# =================================================================
def 원래컬럼(피처):
    for c in 코드컬럼:
        if 피처.startswith(c + "_"):
            return c
    return 피처

이름바꾸기 = {"출생시국적코드(외국)": "출신국", "장애인복지카드유무": "장애인복지카드",
          "해외에서의교육년수": "해외 교육 연수"}
중요도 = pd.Series(best.feature_importances_, index=X.columns)
중요도 = 중요도.groupby(원래컬럼).sum().sort_values(ascending=False)
importance = pd.DataFrame({"feature": [이름바꾸기.get(c, c) for c in 중요도.index],
                           "importance": 중요도.values.round(4)})
importance.to_csv(f"{OUT_DIR}/importance.csv", index=False, encoding="utf-8-sig")
print("\n[특성 중요도]")
print(importance.to_string(index=False))


# =================================================================
# 5. 체류기간별 실제 귀화율 (가중치 적용)
# =================================================================
s = raw.copy()
s["체류기간"] = SURVEY_YEAR - s["최초입국연도_년"]
s = s[s["체류기간"].notna() & (s["체류기간"] >= 0)]
구간 = ["0-2년", "3-5년", "6-8년", "9-11년", "12-15년", "16-20년", "21년+"]
s["구간"] = pd.cut(s["체류기간"], [-1, 2, 5, 8, 11, 15, 20, 90], labels=구간)

rows = []
for i, (이름, g) in enumerate(s.groupby("구간", observed=True)):
    rows.append({"stay_band": 이름, "n": len(g),
                 "actual_rate": round(np.average(g["국적취득여부"], weights=g["가중치"]), 4),
                 "order": i})
rate_by_stay = pd.DataFrame(rows)
rate_by_stay.to_csv(f"{OUT_DIR}/rate_by_stay.csv", index=False, encoding="utf-8-sig")
print("\n[체류기간별 귀화율]")
print(rate_by_stay[["stay_band", "n", "actual_rate"]].to_string(index=False))


# =================================================================
# 6. 출신국 코드별 실제 귀화율 (100명 이상, 인원 상위 12개)
# =================================================================
rows = []
for code, g in raw.groupby("출생시국적코드(외국)"):
    if len(g) >= 100:
        rows.append({"country_code": int(code), "n": len(g),
                     "rate": round(np.average(g["국적취득여부"], weights=g["가중치"]), 4)})
rate_by_country = pd.DataFrame(rows).sort_values("n", ascending=False).head(12)
rate_by_country.to_csv(f"{OUT_DIR}/rate_by_country.csv", index=False, encoding="utf-8-sig")


# =================================================================
# 7. 요약 숫자
# =================================================================
summary = pd.DataFrame([{
    "total": len(raw),
    "naturalized": int((raw["국적취득여부"] == 1).sum()),
    "not_naturalized": int((raw["국적취득여부"] == 0).sum()),
    "train_n": len(X_train), "test_n": len(X_test), "n_features": X.shape[1],
}])
summary.to_csv(f"{OUT_DIR}/summary.csv", index=False, encoding="utf-8-sig")


# =================================================================
# 8. 예측기 입력 선택지 (숫자는 중앙값, 코드는 최빈값이 기본값)
# =================================================================
spec = []
for c, 라벨 in [("최초입국연도_년", "입국연도"), ("만나이", "만 나이"), ("해외에서의교육년수", "해외 교육 연수")]:
    spec.append({"field": c, "label": 라벨, "kind": "number", "default": float(중앙값[c]),
                 "min": float(raw[c].min()), "max": float(raw[c].max()), "code": "", "n": 0})
for c, 라벨 in [("혼인상태", "혼인상태 코드"), ("성별", "성별 코드"), ("출생시국적코드(외국)", "출신국 코드"),
               ("장애인복지카드유무", "장애인복지카드 코드"), ("최초입국여부", "최초입국여부 코드")]:
    for code, n in raw[c].value_counts().items():
        if c == "출생시국적코드(외국)" and n < 30:
            continue                                  # 30명 미만 국가는 '기타'
        spec.append({"field": c, "label": 라벨, "kind": "category", "default": float(raw[c].mode()[0]),
                     "min": 0, "max": 0, "code": float(code), "n": int(n)})
pd.DataFrame(spec).to_csv(f"{OUT_DIR}/input_spec.csv", index=False, encoding="utf-8-sig")


# =================================================================
# 9. 최적 결정트리 구조를 JSON으로 저장 (대시보드 예측기가 읽는 파일)
#    노드마다: 나누는 컬럼(f), 기준값(t), 왼쪽(l)/오른쪽(r) 자식, 귀화 비율(p), 인원(n)
#    규칙: 값 <= 기준값 이면 왼쪽, 아니면 오른쪽. l == -1 이면 잎(마지막 칸)
# =================================================================
t = best.tree_
피처들 = list(X.columns)
nodes = []
for i in range(t.node_count):
    잎 = t.children_left[i] == -1
    v = t.value[i][0]
    nodes.append({"id": i,
                  "f": "" if 잎 else 피처들[t.feature[i]],
                  "t": 0 if 잎 else float(t.threshold[i]),
                  "l": int(t.children_left[i]), "r": int(t.children_right[i]),
                  "p": round(float(v[1] / v.sum()), 4),
                  "n": int(t.n_node_samples[i])})
with open(f"{OUT_DIR}/tree_model.json", "w", encoding="utf-8") as f:
    json.dump(nodes, f, ensure_ascii=False)
print(f"\n결정트리 노드 {len(nodes)}개 저장")


# =================================================================
# 10. 예측기 (대시보드의 '귀화 가능성 예측기'와 같은 계산)
# =================================================================
def 예측하기(입국연도=None, 만나이=None, 해외교육연수=None, 혼인상태=2, 성별=1,
          출신국=185, 장애인복지카드=2, 최초입국여부=1, 경로보기=False):
    """한 사람의 정보를 넣으면 귀화 확률(0~1)을 돌려준다. 빈 숫자는 중앙값으로 채운다."""
    사람 = {
        "최초입국연도_년": 중앙값["최초입국연도_년"] if 입국연도 is None else 입국연도,
        "만나이": 중앙값["만나이"] if 만나이 is None else 만나이,
        "해외에서의교육년수": 중앙값["해외에서의교육년수"] if 해외교육연수 is None else 해외교육연수,
        "혼인상태": 혼인상태, "성별": 성별, "출생시국적코드(외국)": 출신국,
        "장애인복지카드유무": 장애인복지카드, "최초입국여부": 최초입국여부,
    }
    x = pd.DataFrame([사람])
    x["체류기간"] = SURVEY_YEAR - x["최초입국연도_년"]
    x = x.drop(columns=["최초입국연도_년"])
    x[코드컬럼] = x[코드컬럼].astype(float)            # 학습 데이터와 형식 맞추기 (185 → 185.0)
    x = pd.get_dummies(x, columns=코드컬럼)
    x = x.reindex(columns=X.columns, fill_value=0).astype(float)

    확률 = best.predict_proba(x)[0, 1]
    if 경로보기:                                      # 모델이 따라간 판단 경로 출력
        i = 0
        while t.children_left[i] != -1:
            f, 기준 = 피처들[t.feature[i]], t.threshold[i]
            값 = x.iloc[0][f]
            왼쪽 = 값 <= 기준
            print(f"   {f} = {값:g}  {'<=' if 왼쪽 else '>'} {기준:.1f}")
            i = t.children_left[i] if 왼쪽 else t.children_right[i]
    return 확률


if __name__ == "__main__":
    print("\n[예측 예시] 38세, 출신국 185, 혼인상태 2 — 입국연도만 바꾸기")
    for 연도 in [2019, 2015, 2010, 2000]:
        print(f"  {연도}년 입국 → 귀화 확률 {예측하기(입국연도=연도, 만나이=38, 해외교육연수=12):.1%}")

    print("\n[판단 경로] 2010년 입국")
    p = 예측하기(입국연도=2010, 만나이=38, 해외교육연수=12, 경로보기=True)
    print(f"   → 귀화 확률 {p:.1%}")

    print(f"\n저장 위치: {OUT_DIR}/ (CSV 7개 + tree_model.json)")
