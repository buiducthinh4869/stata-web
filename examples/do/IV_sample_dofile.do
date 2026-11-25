///// EXAMPLE 1: IV NHÂN TẠO

clear all
set more off
set seed 2024 // Đặt seed để kết quả giống nhau mỗi lần chạy

* 1. THIẾT LẬP DÂN SỐ (10,000 người)
set obs 10000

* --- Tạo các nhóm ẩn (Latent Groups) ---
* Chúng ta có một biến "Động lực ngầm" (U) không quan sát được.
* U cao = Luôn đi tập (Always-takers). U thấp = Lười (Never-takers).

gen rand_type = runiform()

* Quy định tỷ lệ các nhóm:
* 15% Always-takers (Luôn tập)
* 15% Never-takers  (Không bao giờ tập)
* 70% Compliers     (Nghe lời - Nhóm mục tiêu của IV)

gen Group = .
replace Group = 1 if rand_type <= 0.15          // Always-takers
replace Group = 2 if rand_type > 0.15 & rand_type <= 0.85 // Compliers
replace Group = 3 if rand_type > 0.85           // Never-takers

label define grp_lab 1 "Always-taker" 2 "Complier" 3 "Never-taker"
label values Group grp_lab

* --- Tạo biến nhiễu (Confounder: Sức khỏe nền tảng) ---
* Always-takers thường khỏe sẵn (High base health).
* Never-takers thường yếu sẵn.
gen BaseHealth = .
replace BaseHealth = 80 if Group == 1 // Khỏe sẵn
replace BaseHealth = 60 if Group == 2 // Bình thường
replace BaseHealth = 40 if Group == 3 // Yếu
* Thêm nhiễu ngẫu nhiên vào sức khỏe nền
replace BaseHealth = BaseHealth + rnormal(0, 5)

* 2. PHÂN PHỐI NGẪU NHIÊN (The Experiment)
* Z: Biến công cụ (Voucher). 1 = Có Voucher, 0 = Không.
gen Z_Voucher = (rbinomial(1, 0.5) == 1)

* 3. HÀNH VI ĐI TẬP (Treatment D)
gen D_Gym = 0

* Nếu là Always-taker: Luôn đi tập dù Z=0 hay Z=1
replace D_Gym = 1 if Group == 1 

* Nếu là Never-taker: Không bao giờ tập dù Z=0 hay Z=1
replace D_Gym = 0 if Group == 3

* Nếu là Complier: Đi tập nếu có Voucher (Z=1), Ở nhà nếu không (Z=0)
replace D_Gym = 1 if Group == 2 & Z_Voucher == 1
replace D_Gym = 0 if Group == 2 & Z_Voucher == 0

* 4. TẠO KẾT QUẢ SỨC KHỎE (Outcome Y)
* Giả định: Đi tập Gym giúp tăng THẬT SỰ 10 điểm sức khỏe.
* Y = Sức khỏe nền + (10 * Đi Tập) + Sai số
gen Y_Health = BaseHealth + (10 * D_Gym) + rnormal(0, 2)

* ==========================================================
* PHẦN PHÂN TÍCH VÀ SO SÁNH
* ==========================================================

* --- 1. OLS Naive (So sánh Người đi vs Không đi) ---
* Đây thường được coi là TOT thô sơ (nhưng bị bias).
* Vì nhóm đi tập (D=1) bao gồm cả Always-takers (vốn dĩ đã khỏe),
* nên kết quả sẽ LỚN HƠN 10 (Overestimated).
reg Y_Health D_Gym
est store OLS

* --- 2. ITT (Intention to Treat) ---
* So sánh Nhóm được mời (Z=1) vs Không được mời (Z=0).
* Kết quả sẽ NHỎ HƠN 10 vì bị pha loãng bởi nhóm Never-takers (nhận vé nhưng ko tập).
reg Y_Health Z_Voucher
est store ITT

* Lưu giá trị hệ số ITT để tính tay sau này
scalar b_ITT = _b[Z_Voucher]

* --- 3. First Stage (Giai đoạn 1) ---
* Xem Voucher ảnh hưởng thế nào đến việc đi Gym.
* Hệ số này chính là tỷ lệ tuân thủ (Compliance Rate).
reg D_Gym Z_Voucher
scalar b_FirstStage = _b[Z_Voucher]

* --- 4. LATE (Local Average Treatment Effect) ---
* Phương pháp IV: Dùng 2SLS.
* Kết quả kỳ vọng: Gần bằng 10 (Hiệu quả thực trên nhóm Compliers).
ivregress 2sls Y_Health (D_Gym = Z_Voucher)
est store IV_LATE

* --- Tính tay LATE (Wald Estimator) ---
* Công thức: LATE = ITT / First_Stage
display as text "--------------------------------------------------"
display as text "TÍNH TOÁN THỦ CÔNG (Wald Estimator)"
display as text "ITT Effect (Tác động của Voucher lên Sức khỏe): " as result b_ITT
display as text "First Stage (Tỷ lệ tuân thủ): " as result b_FirstStage
display as text "LATE (ITT / First Stage): " as result b_ITT / b_FirstStage
display as text "TRUE EFFECT (Thiết lập ban đầu): " as result "10"
display as text "--------------------------------------------------"

* --- Bảng tổng hợp ---
estimates table OLS ITT IV_LATE, star stats(N) b(%9.3f) title("So sánh các mô hình")



/// EXAMPLE 2: IV TỰ NHIÊN



* 1. THIẾT LẬP VÀ MÔ PHỎNG DỮ LIỆU (DGP)
*----------------------------------------------------------
clear all
set more off
set seed 20251206 // "Con số may mắn" của lớp
set obs 5000

* 1.1. Tạo biến "Năng lực" (Ability - Biến ẩn/Unobserved)
* Đây là mấu chốt: Người giỏi (Ability cao) thì vừa học cao, vừa lương cao.
* Nhưng ta KHÔNG đo lường được Ability trong thực tế -> Gây nội sinh.
gen ability = rnormal(0, 1)

* 1.2. Tạo Biến công cụ Z: "Khoảng cách từ nhà đến trường ĐH" (Distance)
* Giả định: Nhà xa trường thì lười đi học ĐH hơn.
* Khoảng cách là ngẫu nhiên (Exogenous), không liên quan đến Năng lực.
gen distance = rchi2(3) 
label var distance "Khoảng cách đến trường ĐH (Biến công cụ Z)"

* 1.3. GIAI ĐOẠN 1 (First Stage): Tạo biến X (Số năm đi học)
* Học vấn phụ thuộc vào: 
* (+) Năng lực (Ability)
* (-) Khoảng cách (Distance) - Càng xa càng ngại đi học
gen educ = 10 + 1.5*ability - 0.8*distance + rnormal(0, 2)
label var educ "Số năm đi học (Biến nội sinh X)"

* 1.4. GIAI ĐOẠN 2 (Structural Eq): Tạo biến Y (Thu nhập - Log Wage)
* Tác động thực (TRUE RETURN) của giáo dục là: 10% (0.10)
* Lương phụ thuộc vào: Học vấn + Năng lực (Biến ẩn)
local true_beta = 0.10
gen ln_wage = 5.0 + `true_beta'*educ + 0.05*ability + rnormal(0, 0.5)
label var ln_wage "Log Thu nhập (Outcome Y)"

* -> GIẢI THÍCH CHO SINH VIÊN:
* Mô hình thực tế chúng ta có số liệu là: ln_wage = alpha + beta*educ + epsilon
* Nhưng epsilon chứa "ability".
* Do educ tăng theo ability, và wage cũng tăng theo ability -> OLS sẽ bị Ước lượng quá cao (Overestimation).

*----------------------------------------------------------
* 2. CHẠY HỒI QUY OLS (ĐỂ THẤY SỰ THIÊN LỆCH)
*----------------------------------------------------------
display "--------------------------------------------------"
display "KẾT QUẢ 1: HỒI QUY OLS THÔNG THƯỜNG (BỊ CHỆCH)"
display "True Beta = `true_beta' (0.10)"
display "--------------------------------------------------"
reg ln_wage educ

* -> Hướng dẫn đọc:
* Hệ số của 'educ' sẽ khoảng 0.12 - 0.13 (Cao hơn 0.10).
* Kết luận: OLS bị chệch do thiếu biến (Omitted Variable Bias).

*----------------------------------------------------------
* 3. CHẠY HỒI QUY IV (2SLS) - PHƯƠNG PHÁP 2 GIAI ĐOẠN
*----------------------------------------------------------
* Cú pháp: ivregress 2sls biếny (biếnx = biếnz)
display "--------------------------------------------------"
display "KẾT QUẢ 2: HỒI QUY IV (2SLS) - ĐÃ HIỆU CHỈNH"
display "--------------------------------------------------"

ivregress 2sls ln_wage (educ = distance)

* -> Hướng dẫn đọc:
* Nhìn hệ số 'educ': Nó sẽ quay về rất gần 0.10 (True Effect).
* Khoảng tin cậy (Confidence Interval) của IV thường rộng hơn OLS (do trade-off về phương sai).

*----------------------------------------------------------
* 4. CÁC KIỂM ĐỊNH QUAN TRỌNG (DIAGNOSTICS)
*----------------------------------------------------------
* IV không phải cứ chạy là đúng, phải kiểm tra các điều kiện.

* 4.1. Kiểm tra độ mạnh của biến công cụ (Weak Instrument Test)
* Z phải tác động mạnh lên X thì IV mới tin cậy.
display ">>> KIỂM TRA GIAI ĐOẠN 1 (FIRST STAGE) <<<"
estat firststage

* -> Hướng dẫn đọc:
* Nhìn vào F-statistic.
* Quy tắc ngón tay cái (Rule of thumb): F > 10 là biến công cụ mạnh.
* Nếu F < 10: "Weak Instrument" -> Kết quả IV sai lệch nghiêm trọng.

* 4.2. Kiểm định tính nội sinh (Test of Endogeneity)
* Kiểm tra xem "Liệu biến educ có thực sự bị nội sinh không?".
* Nếu nó không nội sinh thì cứ dùng OLS cho hiệu quả (với phương sai nhỏ hơn).
* Null hypothesis (H0): Biến X là ngoại sinh.
display ">>> KIỂM ĐỊNH NỘI SINH (DURBIN-WU-HAUSMAN) <<<"
estat endogenous

* -> Hướng dẫn đọc:
* Nhìn Robust score chi2 hoặc Robust regression F.
* P-value < 0.05: Bác bỏ H0 -> Kết luận biến educ LÀ nội sinh -> Bắt buộc phải dùng IV.

*----------------------------------------------------------
* MINH HỌA TRỰC QUAN QUAN HỆ CỦA BIẾN CÔNG CỤ (OPTIONAL)
*----------------------------------------------------------
* Vẽ biểu đồ để thấy Z tác động lên X như thế nào (First Stage Relationship)
twoway (scatter educ distance, msize(tiny) mcolor(gray%30)) ///
       (lfit educ distance, lcolor(red) lwidth(thick)), ///
       title("Giai đoạn 1: Mối quan hệ giữa Khoảng cách (Z) và Học vấn (X)") ///
       xtitle("Khoảng cách đến trường (IV)") ytitle("Số năm đi học (Endogenous)") ///
       legend(off)
///// graph export "C:\Users\Windows 10\Downloads\Graph.png", as(png) name("Graph")
*=============================================================================
* KẾT THÚC BÀI GIẢNG IV
*=============================================================================