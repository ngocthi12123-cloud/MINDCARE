import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Đặt file CSV cùng thư mục với file này rồi chạy:
#   streamlit run mindcare_app.py
# ─────────────────────────────────────────────────────────────────────────────
CSV_FILE = "StressLevelDataset.csv"

# ─── Cấu hình trang ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindCare - Dự đoán Căng thẳng & Trầm cảm",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
/* Nền tối như app Replit */
.stApp { background-color: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

/* Tiêu đề */
h1 { color: #e6edf3; font-size: 2rem; }
h2 { color: #e6edf3; }
h3 { color: #58a6ff; font-size: 1.1rem; margin-top: 1.2rem; }

/* Card / metric */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Badge style cho impact */
.badge-high   { background:#3d1717; color:#f85149; border:1px solid #f85149;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#2d2007; color:#e3b341; border:1px solid #e3b341;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-low    { background:#0d1e2e; color:#58a6ff; border:1px solid #58a6ff;
                border-radius:6px; padding:2px 10px; font-size:0.78rem; font-weight:600; }

/* Nút primary */
div[data-testid="stButton"] button[kind="primary"] {
    background: #238636; color: #fff; border: none;
    border-radius: 8px; font-size: 1rem; padding: 0.65rem 2rem;
    width: 100%;
}
div[data-testid="stButton"] button[kind="primary"]:hover { background: #2ea043; }

/* Lời khuyên */
.rec-item {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
    display: flex; gap: 10px; align-items: flex-start;
}
.rec-dot { color: #238636; font-size: 1.2rem; line-height: 1.4; }

/* Progress bar custom */
.prog-bg  { background:#21262d; border-radius:6px; height:8px; margin:4px 0 12px; }
.prog-fill-high   { background:#f85149; height:8px; border-radius:6px; }
.prog-fill-medium { background:#e3b341; height:8px; border-radius:6px; }
.prog-fill-low    { background:#58a6ff; height:8px; border-radius:6px; }
.prog-fill-green  { background:#238636; height:8px; border-radius:6px; }
.prog-fill-prim   { background:#58a6ff; height:8px; border-radius:6px; }

/* Expander */
details { background:#161b22; border:1px solid #30363d; border-radius:8px;
          margin-bottom:8px; padding:4px 8px; }
summary { color:#e6edf3; font-weight:600; font-size:1rem; cursor:pointer; }

div[data-testid="stExpander"] { border:1px solid #30363d; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ─── Hằng số ──────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "anxiety_level","self_esteem","mental_health_history","depression",
    "headache","blood_pressure","sleep_quality","breathing_problem",
    "noise_level","living_conditions","safety","basic_needs",
    "academic_performance","study_load","teacher_student_relationship",
    "future_career_concerns","social_support","peer_pressure",
    "extracurricular_activities","bullying",
]

FEATURE_MAX = {
    "anxiety_level":21,"self_esteem":30,"mental_health_history":1,"depression":27,
    "headache":5,"blood_pressure":3,"sleep_quality":5,"breathing_problem":5,
    "noise_level":5,"living_conditions":5,"safety":5,"basic_needs":5,
    "academic_performance":5,"study_load":5,"teacher_student_relationship":5,
    "future_career_concerns":5,"social_support":5,"peer_pressure":5,
    "extracurricular_activities":5,"bullying":5,
}

FEATURE_VI = {
    "anxiety_level":               ("Mức độ lo lắng",            "Bạn cảm thấy lo âu như thế nào gần đây?",      0,21,10),
    "self_esteem":                 ("Lòng tự trọng",             "Mức độ tự tin và tôn trọng bản thân.",         0,30,15),
    "mental_health_history":       ("Tiền sử sức khỏe tâm thần","Gia đình hoặc bạn từng có vấn đề tâm lý?",    0, 1, 0),
    "depression":                  ("Mức độ trầm cảm",           "Cảm giác buồn bã, mất hứng thú.",             0,27,10),
    "headache":                    ("Tần suất đau đầu",          "Tần suất bạn bị đau đầu.",                    0, 5, 2),
    "blood_pressure":              ("Huyết áp",                  "1: Thấp  2: Bình thường  3: Cao.",            1, 3, 2),
    "sleep_quality":               ("Chất lượng giấc ngủ",       "Mức độ ngủ ngon và đủ giấc.",                 0, 5, 3),
    "breathing_problem":           ("Vấn đề hô hấp",             "Khó thở, hụt hơi.",                          0, 5, 1),
    "noise_level":                 ("Mức độ tiếng ồn",           "Nơi ở/học tập của bạn có ồn ào không?",      0, 5, 2),
    "living_conditions":           ("Điều kiện sống",            "Chất lượng không gian sống.",                 0, 5, 3),
    "safety":                      ("Cảm giác an toàn",          "Bạn cảm thấy an toàn ở khu vực mình sống?",  0, 5, 3),
    "basic_needs":                 ("Nhu cầu cơ bản",            "Mức độ đáp ứng ăn, uống, sinh hoạt.",        0, 5, 3),
    "academic_performance":        ("Kết quả học tập",           "Điểm số, thành tích học tập.",                0, 5, 3),
    "study_load":                  ("Áp lực học tập",            "Khối lượng bài vở, thi cử.",                  0, 5, 3),
    "teacher_student_relationship":("Quan hệ thầy trò",          "Mức độ thân thiện, hỗ trợ từ giáo viên.",    0, 5, 3),
    "future_career_concerns":      ("Lo lắng tương lai",         "Căng thẳng về định hướng nghề nghiệp.",       0, 5, 3),
    "social_support":              ("Hỗ trợ xã hội",             "Gia đình, bạn bè có sẵn sàng giúp đỡ bạn?", 0, 5, 3),
    "peer_pressure":               ("Áp lực từ bạn bè",          "Cảm giác phải bằng bạn bằng bè.",            0, 5, 2),
    "extracurricular_activities":  ("Hoạt động ngoại khóa",      "Mức độ tham gia CLB, thể thao.",              0, 5, 2),
    "bullying":                    ("Bắt nạt",                   "Bị trêu chọc, bạo lực học đường/mạng.",      0, 5, 0),
}

GROUPS = {
    "🧠 Tâm lý (Psychological)":   ["anxiety_level","self_esteem","mental_health_history","depression"],
    "💪 Thể chất (Physical)":      ["headache","blood_pressure","sleep_quality","breathing_problem"],
    "🏘️ Môi trường (Environmental)":["noise_level","living_conditions","safety","basic_needs"],
    "📚 Học tập (Academic)":       ["academic_performance","study_load","teacher_student_relationship","future_career_concerns"],
    "👥 Xã hội (Social)":          ["social_support","peer_pressure","extracurricular_activities","bullying"],
}

STRESS_LABELS  = {0:"Thấp", 1:"Trung bình", 2:"Cao"}
STRESS_COLORS  = {0:"#39d353", 1:"#e3b341", 2:"#f85149"}

RECOMMENDATIONS = {
    0:[
        "Tiếp tục duy trì lối sống lành mạnh và cân bằng.",
        "Tham gia các hoạt động thể thao và ngoại khóa để tăng cường sức khỏe.",
        "Duy trì kết nối xã hội tích cực với bạn bè và gia đình.",
        "Ngủ đủ giấc từ 7-9 tiếng mỗi đêm.",
        "Thực hành chánh niệm (mindfulness) 10-15 phút mỗi ngày.",
    ],
    1:[
        "Hãy dành thời gian nghỉ ngơi và thư giãn mỗi ngày.",
        "Chia sẻ cảm xúc với người thân hoặc bạn bè đáng tin cậy.",
        "Lập kế hoạch học tập hợp lý, tránh làm việc quá sức.",
        "Tập thể dục nhẹ nhàng như đi bộ, yoga ít nhất 30 phút/ngày.",
        "Hạn chế caffeine và duy trì giờ ngủ đều đặn.",
        "Cân nhắc tham khảo ý kiến chuyên gia tâm lý nếu cảm thấy cần thiết.",
    ],
    2:[
        "Hãy tìm kiếm sự hỗ trợ từ chuyên gia tâm lý hoặc bác sĩ ngay.",
        "Chia sẻ tình trạng của bạn với người thân để được hỗ trợ.",
        "Không tự mình chịu đựng — bạn không phải chiến đấu một mình.",
        "Ưu tiên sức khỏe tâm thần hơn các áp lực học tập hoặc công việc.",
        "Thực hành các kỹ thuật thở sâu và thư giãn cơ thể.",
        "Tránh cô lập bản thân — duy trì kết nối với mọi người xung quanh.",
        "Đặt các mục tiêu nhỏ, dễ đạt được để tạo cảm giác thành công.",
    ],
}

# ─── Huấn luyện mô hình ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang huấn luyện mô hình học máy...")
def load_and_train():
    df = pd.read_csv(CSV_FILE)
    X = df[FEATURE_NAMES].values
    y = df["stress_level"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_tr, y_train)

    acc    = accuracy_score(y_test, model.predict(X_te))
    report = classification_report(y_test, model.predict(X_te),
                                   target_names=["Thấp","Trung bình","Cao"])

    # Phân bố class
    counts = [(y == c).sum() for c in [0,1,2]]
    total  = len(y)
    class_dist = [
        {"label": STRESS_LABELS[c], "count": counts[c],
         "percentage": round(counts[c]/total*100, 1)}
        for c in [0,1,2]
    ]

    return model, scaler, df, acc, report, class_dist


# ─── Predict ──────────────────────────────────────────────────────────────────
def predict(model, scaler, inp: dict):
    feat = np.array([[inp[f] for f in FEATURE_NAMES]])
    feat_s = scaler.transform(feat)
    stress_level = int(model.predict(feat_s)[0])
    proba = model.predict_proba(feat_s)[0]   # [p_low, p_med, p_high]

    # Gauge value (0-100) — giống hệt Replit
    if stress_level == 0:
        stress_gauge = (1 - proba[0]) * 33
    elif stress_level == 1:
        stress_gauge = 33 + proba[1] * 33
    else:
        stress_gauge = 66 + proba[2] * 34

    # Anxiety score
    anxiety_score = min(100, round(
        (inp["anxiety_level"]/21)*40 +
        (inp["depression"]/27)*20 +
        ((5-inp["sleep_quality"])/5)*20 +
        (inp["peer_pressure"]/5)*10 +
        (inp["future_career_concerns"]/5)*10
    ))

    # Depression score
    depression_score = min(100, round(
        (inp["depression"]/27)*40 +
        ((30-inp["self_esteem"])/30)*25 +
        ((5-inp["social_support"])/5)*15 +
        (inp["bullying"]/5)*10 +
        ((5-inp["basic_needs"])/5)*10
    ))

    # Risk factors — dùng đúng logic ml-model.ts
    coef_high = model.coef_[2]   # weights cho class "Cao"
    risk_factors = []
    for i, name in enumerate(FEATURE_NAMES):
        norm_val = inp[name] / FEATURE_MAX[name] if FEATURE_MAX[name] else 0
        w = coef_high[i]
        abs_w = abs(w)
        if abs_w <= 0.05:
            continue
        impact = "high" if abs_w > 0.5 else "medium" if abs_w > 0.2 else "low"
        risk_factors.append({
            "name": name,
            "name_vi": FEATURE_VI[name][0],
            "value": inp[name],
            "weight": round(abs_w, 3),
            "impact": impact,
        })
    risk_factors = sorted(risk_factors, key=lambda x: x["weight"], reverse=True)[:5]

    return {
        "stress_level": stress_level,
        "stress_label": STRESS_LABELS[stress_level],
        "confidence_low": round(float(proba[0]), 3),
        "confidence_medium": round(float(proba[1]), 3),
        "confidence_high": round(float(proba[2]), 3),
        "stress_gauge": stress_gauge,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "risk_factors": risk_factors,
        "recommendations": RECOMMENDATIONS[stress_level],
    }


# ─── Vẽ đồng hồ gauge (SVG-style bằng matplotlib) ─────────────────────────────
def draw_gauge(value: float, title: str, subtitle: str, color: str):
    fig, ax = plt.subplots(figsize=(3.4, 2.2), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    # Track nền
    theta_bg = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg),
            color="#21262d", lw=16, solid_capstyle="round")

    # Track giá trị
    pct = max(0.01, min(1.0, value / 100))
    end_angle = np.pi - pct * np.pi
    theta_v = np.linspace(np.pi, end_angle, max(3, int(pct * 300)))
    ax.plot(np.cos(theta_v), np.sin(theta_v),
            color=color, lw=16, solid_capstyle="round")

    # Kim đồng hồ
    needle_angle = np.pi - pct * np.pi
    ax.plot([0, 0.60 * np.cos(needle_angle)],
            [0, 0.60 * np.sin(needle_angle)],
            color="#e6edf3", lw=2.5, zorder=5)
    ax.plot(0, 0, "o", color="#e6edf3", ms=7, zorder=6)

    # Số giữa
    ax.text(0, -0.18, str(int(value)), ha="center", va="center",
            fontsize=20, fontweight="bold", color=color)
    # Label
    ax.text(0, -0.38, subtitle, ha="center", va="center",
            fontsize=7.5, color="#8b949e")
    ax.text(0,  1.05, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color="#e6edf3")
    # Min / Max
    ax.text(-1.08, -0.10, "0",   ha="center", color="#8b949e", fontsize=7)
    ax.text( 1.08, -0.10, "100", ha="center", color="#8b949e", fontsize=7)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.55, 1.25)
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    return fig


# ─── Thanh progress có màu theo impact ────────────────────────────────────────
def render_risk_factor(rf):
    impact  = rf["impact"]
    color   = "#f85149" if impact=="high" else "#e3b341" if impact=="medium" else "#58a6ff"
    badge_c = "badge-high" if impact=="high" else "badge-medium" if impact=="medium" else "badge-low"
    label_i = "Ảnh hưởng cao" if impact=="high" else "Ảnh hưởng vừa" if impact=="medium" else "Ảnh hưởng thấp"
    pct     = min(100, int(rf["weight"] * 100))
    fill_c  = "prog-fill-high" if impact=="high" else "prog-fill-medium" if impact=="medium" else "prog-fill-low"

    st.markdown(f"""
    <div style="margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-weight:600;color:#e6edf3">{rf['name_vi']}</span>
        <span class="{badge_c}">{label_i}</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px">
        <div class="prog-bg" style="flex:1">
          <div class="{fill_c}" style="width:{pct}%"></div>
        </div>
        <span style="color:#8b949e;font-size:0.8rem;font-family:monospace;min-width:40px;text-align:right">{rf['value']} pt</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    model, scaler, df, acc, report, class_dist = load_and_train()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 MindCare")
        st.markdown(
            "Ứng dụng dự đoán mức độ **căng thẳng & trầm cảm** bằng học máy "
            "dành cho học sinh, sinh viên Việt Nam."
        )
        st.divider()
        page = st.radio(
            "Điều hướng",
            ["🏠 Trang chủ", "📋 Đánh giá", "📊 Thông tin mô hình"],
        )

    # ══ TRANG CHỦ ═════════════════════════════════════════════════════════════
    if page == "🏠 Trang chủ":
        # Hero
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0d2137,#0d1117);
                    border:1px solid #30363d;border-radius:12px;
                    padding:40px 32px;margin-bottom:32px;text-align:center">
          <div style="display:inline-flex;align-items:center;gap:8px;
                      background:#0d2137;border:1px solid #388bfd44;
                      border-radius:20px;padding:4px 14px;
                      color:#58a6ff;font-size:0.85rem;margin-bottom:16px">
            🤖 AI-Powered Mental Health Assessment
          </div>
          <h1 style="font-size:2.4rem;margin:0 0 12px">
            Hiểu rõ tâm trí bạn với <span style="color:#238636">MindCare</span>
          </h1>
          <p style="color:#8b949e;font-size:1.05rem;max-width:560px;margin:0 auto">
            Ứng dụng đánh giá mức độ căng thẳng, lo âu và trầm cảm dành riêng
            cho học sinh, sinh viên Việt Nam — dựa trên mô hình học máy phân tích
            <strong style="color:#e6edf3">20 yếu tố rủi ro</strong>.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # 2 card: Dataset + Model
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown("""
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:10px;padding:24px">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                <span style="font-size:1.1rem">🗃️</span>
                <span style="font-weight:700;font-size:1rem;color:#e6edf3">Dữ liệu huấn luyện</span>
              </div>
              <p style="color:#8b949e;font-size:0.85rem;margin:0 0 20px">
                Bộ dữ liệu thực tế từ cộng đồng sinh viên
              </p>
            """, unsafe_allow_html=True)

            st.markdown(f"""
              <div style="margin-bottom:20px">
                <p style="font-size:2.2rem;font-weight:700;color:#e6edf3;margin:0">{len(df):,}</p>
                <p style="color:#8b949e;font-size:0.85rem;margin:0">Tổng số mẫu khảo sát</p>
              </div>
              <p style="font-size:0.85rem;font-weight:600;color:#e6edf3;margin-bottom:10px">Phân bổ dữ liệu:</p>
            """, unsafe_allow_html=True)

            for item in class_dist:
                color = STRESS_COLORS[[0,1,2][["Thấp","Trung bình","Cao"].index(item["label"])]]
                st.markdown(f"""
                <div style="margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="color:#8b949e;font-size:0.85rem">{item['label']}</span>
                    <span style="font-weight:600;font-size:0.85rem">{item['percentage']}%</span>
                  </div>
                  <div class="prog-bg">
                    <div style="background:{color};height:8px;border-radius:6px;width:{item['percentage']}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("""
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:10px;padding:24px">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                <span style="font-size:1.1rem">⚡</span>
                <span style="font-weight:700;font-size:1rem;color:#e6edf3">Mô hình học máy</span>
              </div>
              <p style="color:#8b949e;font-size:0.85rem;margin:0 0 20px">
                Sử dụng thuật toán phân loại độ chính xác cao
              </p>
            """, unsafe_allow_html=True)

            st.markdown(f"""
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px">
                <div style="background:#0d1117;border:1px solid #30363d;
                            border-radius:8px;padding:16px">
                  <p style="font-size:1.8rem;font-weight:700;color:#e6edf3;margin:0">{acc*100:.1f}%</p>
                  <p style="color:#8b949e;font-size:0.82rem;margin:0">Độ chính xác</p>
                </div>
                <div style="background:#0d1117;border:1px solid #30363d;
                            border-radius:8px;padding:16px">
                  <p style="font-size:1.8rem;font-weight:700;color:#e6edf3;margin:0">20</p>
                  <p style="color:#8b949e;font-size:0.82rem;margin:0">Yếu tố phân tích</p>
                </div>
              </div>
              <p style="font-size:0.85rem;font-weight:600;color:#e6edf3;margin-bottom:10px">Thuật toán sử dụng:</p>
              <div style="display:flex;align-items:center;gap:8px">
                <span style="color:#238636;font-size:1rem">✔</span>
                <span style="color:#8b949e;font-size:0.9rem">Multinomial Logistic Regression (Softmax)</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ══ ĐÁNH GIÁ ══════════════════════════════════════════════════════════════
    elif page == "📋 Đánh giá":
        st.markdown("## Đánh giá Tâm lý")
        st.markdown(
            "<p style='color:#8b949e'>Hãy trả lời trung thực các yếu tố dưới đây để hệ thống "
            "phân tích tình trạng của bạn. Kéo thanh trượt để chọn mức độ phù hợp.</p>",
            unsafe_allow_html=True,
        )
        # ─── ĐOẠN THÊM VÀO: Hộp hướng dẫn đọc thang điểm ───
        with st.expander("💡 Hướng dẫn về thang điểm (Dành cho người dùng)", expanded=False):
             st.markdown("""
        Để đảm bảo tính chính xác khoa học của mô hình học máy, các con số được giữ theo thang đo tâm lý chuẩn:
        * **Mức độ lo lắng (0–21 pt):** 0-4: Bình thường | 5-9: Nhẹ | 10-14: Vừa | 15-21: Nặng
        * **Mức độ trầm cảm (0–27 pt):** 0-4: Bình thường / Tối thiểu | 5-9: Nhẹ | 10-14: Vừa (Trung bình) | 15-19: Khá nặng | 20-27: Nặng
        * **Lòng tự trọng (0–30 pt):** 0-14: Thấp (Tự ti) | 15-25: Bình thường / Khỏe mạnh | 26-30: Cao (Rất tự tin)
        * **Các yếu tố khác (0–5 pt):** 0-1: Thấp / Hiếm khi | 2-3: Trung bình / Thỉnh thoảng | 4-5: Cao / Thường xuyên.
        
        *Hệ thống sẽ tự động quy đổi và chuẩn hóa các con số này trước khi đưa vào AI phân tích, bạn chỉ cần chọn mức độ đúng với cảm nhận hiện tại của mình.*
        """)

        inp = {}
        for grp_label, feats in GROUPS.items():
            with st.expander(grp_label, expanded=True):
                for f in feats:
                    label_vi, desc, lo, hi, default = FEATURE_VI[f]

                    if f == "mental_health_history":
                        # Toggle như app Replit
                        checked = st.toggle(
                            f"**{label_vi}**",
                            value=bool(default),
                            help=desc,
                            key=f,
                        )
                        inp[f] = 1 if checked else 0
                        st.caption(desc)
                    else:
                        col_l2, col_r2 = st.columns([5, 1])
                        with col_l2:
                            st.markdown(f"**{label_vi}**")
                            if desc:
                                st.caption(desc)
                        with col_r2:
                            st.markdown(
                                f"<div style='background:#0d2137;color:#58a6ff;"
                                f"font-weight:700;font-size:1.2rem;padding:4px 12px;"
                                f"border-radius:6px;text-align:center;margin-top:4px'>"
                                f"<span id='val_{f}'>-</span></div>",
                                unsafe_allow_html=True,
                            )
                        val = st.slider(
                            f"{label_vi} ({lo}–{hi})",
                            min_value=lo, max_value=hi, value=default,
                            key=f, label_visibility="collapsed",
                        )
                        inp[f] = val
                        # Hiện giá trị thực
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"color:#8b949e;font-size:0.78rem;margin-top:-8px;margin-bottom:8px'>"
                            f"<span>Thấp ({lo})</span>"
                            f"<span style='color:#58a6ff;font-weight:600'>{val}</span>"
                            f"<span>Cao ({hi})</span></div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("<hr style='border:none;border-top:1px solid #21262d;margin:8px 0'>",
                                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Card submit
        st.markdown("""
        <div style="background:#0d2137;border:1px solid #238636;
                    border-radius:10px;padding:20px 24px;margin-bottom:16px">
          <p style="font-weight:700;color:#e6edf3;margin:0 0 4px">Hoàn tất đánh giá</p>
          <p style="color:#8b949e;font-size:0.87rem;margin:0">
            Hãy kiểm tra lại các thông tin trước khi yêu cầu AI phân tích.
          </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍  Phân tích & Dự đoán", type="primary"):
            result = predict(model, scaler, inp)
            st.session_state["result"] = result
            st.session_state["inp"]    = inp

        # Hiển thị kết quả ngay bên dưới nếu đã có
        if "result" in st.session_state:
            _render_results(st.session_state["result"])

    # ══ THÔNG TIN MÔ HÌNH ══════════════════════════════════════════════════════
    elif page == "📊 Thông tin mô hình":
        st.markdown("## Thông tin Mô hình Học máy")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Thuật toán",        "Logistic Regression")
        c2.metric("Độ chính xác",      f"{acc*100:.1f}%")
        c3.metric("Mẫu huấn luyện",    f"{len(df):,}")
        c4.metric("Số lớp đầu ra",     "3")

        st.divider()

        # Báo cáo phân loại
        st.markdown("### 📄 Báo cáo phân loại chi tiết")
        st.code(report, language="text")

        # Biểu đồ trọng số
        st.markdown("### ⚖️ Trọng số đặc trưng (ảnh hưởng đến mức Cao)")
        coef_df = pd.DataFrame({
            "Đặc trưng": [FEATURE_VI[f][0] for f in FEATURE_NAMES],
            "Trọng số":  model.coef_[2].round(3),
        }).sort_values("Trọng số", ascending=False)

        fig_c, ax_c = plt.subplots(figsize=(9, 5))
        fig_c.patch.set_facecolor("#161b22"); ax_c.set_facecolor("#161b22")
        bar_colors = ["#f85149" if v > 0 else "#39d353" for v in coef_df["Trọng số"]]
        ax_c.barh(coef_df["Đặc trưng"], coef_df["Trọng số"],
                  color=bar_colors, edgecolor="#30363d")
        ax_c.axvline(0, color="#8b949e", lw=0.8)
        ax_c.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax_c.spines.values(): sp.set_color("#30363d")
        ax_c.set_xlabel("Trọng số", color="#8b949e")
        ax_c.legend(
            handles=[
                mpatches.Patch(color="#f85149", label="Tăng căng thẳng"),
                mpatches.Patch(color="#39d353", label="Giảm căng thẳng"),
            ],
            facecolor="#161b22", labelcolor="#e6edf3", fontsize=8,
        )
        plt.tight_layout()
        st.pyplot(fig_c); plt.close(fig_c)

        # Boxplot theo mức
        st.markdown("### 📦 Phân phối đặc trưng theo mức độ căng thẳng")
        sel = st.selectbox(
            "Chọn đặc trưng:",
            options=FEATURE_NAMES,
            format_func=lambda f: FEATURE_VI[f][0],
        )
        fig_b, ax_b = plt.subplots(figsize=(6, 3.5))
        fig_b.patch.set_facecolor("#161b22"); ax_b.set_facecolor("#161b22")
        data_bc = [df[df["stress_level"]==c][sel].values for c in [0,1,2]]
        bp = ax_b.boxplot(data_bc, labels=["Thấp","Trung bình","Cao"],
                          patch_artist=True, medianprops=dict(color="#e6edf3", lw=2))
        for patch, col in zip(bp["boxes"], ["#39d353","#e3b341","#f85149"]):
            patch.set_facecolor(col+"44"); patch.set_edgecolor(col)
        ax_b.tick_params(colors="#8b949e")
        for sp in ax_b.spines.values(): sp.set_color("#30363d")
        ax_b.set_ylabel(FEATURE_VI[sel][0], color="#8b949e")
        ax_b.set_xlabel("Mức độ căng thẳng", color="#8b949e")
        plt.tight_layout()
        st.pyplot(fig_b); plt.close(fig_b)


# ─── Render trang kết quả ─────────────────────────────────────────────────────
def _render_results(result):
    st.markdown("---")
    stress_level = result["stress_level"]
    color = STRESS_COLORS[stress_level]

    # Banner kết quả
    bg = {"0":"#0d1e10","1":"#1e1a0d","2":"#1e0d0d"}[str(stress_level)]
    border = {"0":"#238636","1":"#e3b341","2":"#f85149"}[str(stress_level)]
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:10px;
                padding:20px 24px;margin:16px 0;text-align:center">
      <p style="color:#8b949e;font-size:0.9rem;margin:0 0 4px">Kết quả Phân tích — MindCare</p>
      <p style="font-size:1.8rem;font-weight:700;color:{color};margin:0">
        Mức độ căng thẳng: {result['stress_label']}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 3 gauge
    st.markdown("### Chỉ số sức khỏe tâm thần")
    g1, g2, g3 = st.columns(3)

    with g1:
        fig1 = draw_gauge(result["stress_gauge"], "Mức độ Căng thẳng",
                          "Chỉ số Tổng hợp", color)
        st.pyplot(fig1); plt.close(fig1)

    with g2:
        a_score = result["anxiety_score"]
        a_col   = "#39d353" if a_score < 34 else "#e3b341" if a_score < 67 else "#f85149"
        fig2 = draw_gauge(a_score, "Mức độ Lo âu", "Anxiety Score", a_col)
        st.pyplot(fig2); plt.close(fig2)

    with g3:
        d_score = result["depression_score"]
        d_col   = "#39d353" if d_score < 34 else "#e3b341" if d_score < 67 else "#f85149"
        fig3 = draw_gauge(d_score, "Mức độ Trầm cảm", "Depression Score", d_col)
        st.pyplot(fig3); plt.close(fig3)

    # Xác suất
    st.markdown("### Xác suất dự đoán theo mức")
    p_cols = st.columns(3)
    conf_data = [
        ("Thấp",     result["confidence_low"],    "#39d353"),
        ("Trung bình",result["confidence_medium"],"#e3b341"),
        ("Cao",      result["confidence_high"],   "#f85149"),
    ]
    for col, (lbl, pct, clr) in zip(p_cols, conf_data):
        with col:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:16px;text-align:center">
              <p style="font-size:1.6rem;font-weight:700;color:{clr};margin:0">{pct*100:.1f}%</p>
              <p style="color:#8b949e;font-size:0.85rem;margin:4px 0 10px">{lbl}</p>
              <div class="prog-bg">
                <div style="background:{clr};height:8px;border-radius:6px;width:{pct*100:.1f}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_risk, col_rec = st.columns(2, gap="large")

    # Yếu tố rủi ro
    with col_risk:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
          <span style="color:#f85149">⚠️</span>
          <span style="font-weight:700;font-size:1rem;color:#e6edf3">Yếu tố Rủi ro Hàng đầu</span>
        </div>
        <p style="color:#8b949e;font-size:0.85rem;margin-bottom:16px">
          Các yếu tố ảnh hưởng mạnh nhất đến tình trạng của bạn
        </p>
        """, unsafe_allow_html=True)

        for rf in result["risk_factors"]:
            render_risk_factor(rf)

        if not result["risk_factors"]:
            st.markdown("<p style='color:#8b949e;text-align:center;padding:24px 0'>Không tìm thấy yếu tố rủi ro đáng kể.</p>",
                        unsafe_allow_html=True)

    # Lời khuyên
    with col_rec:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
          <span style="color:#58a6ff">💡</span>
          <span style="font-weight:700;font-size:1rem;color:#e6edf3">Lời khuyên Cá nhân hóa</span>
        </div>
        <p style="color:#8b949e;font-size:0.85rem;margin-bottom:16px">
          Gợi ý từ hệ thống để cải thiện tình trạng của bạn
        </p>
        """, unsafe_allow_html=True)

        for rec in result["recommendations"]:
            st.markdown(f"""
            <div class="rec-item">
              <span class="rec-dot">⚡</span>
              <span style="color:#e6edf3;line-height:1.6">{rec}</span>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
