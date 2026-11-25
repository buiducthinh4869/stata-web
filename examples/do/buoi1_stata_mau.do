/* ============================================================
   DO FILE MẪU – BUỔI 1: QUẢN LÝ & KHÁM PHÁ DỮ LIỆU
   Khóa học: Stata cho Nghiên cứu Kinh tế (2 buổi)
   Giảng viên: TS. Đỗ Ngọc Kiên
   Bộ môn Kinh tế & Quản lý – ĐH Ngoại Thương
   
   Mô tả: Do file này sử dụng dữ liệu MÔ PHỎNG gồm 63 tỉnh/thành
   Việt Nam trong giai đoạn 2015–2022 (dữ liệu bảng cân bằng).
   Biến số gồm: GDP bình quân đầu người, FDI, thương mại, dân số,
   vùng kinh tế và khu vực kinh tế.
   ============================================================ */


/* ============================================================
   PHẦN 0: KHỞI TẠO VÀ TẠO DỮ LIỆU MÔ PHỎNG
   ============================================================ */

* --- 0.1 Dọn dẹp môi trường làm việc ---
clear all          // Xóa dữ liệu và kết quả cũ
set more off       // Không dừng khi output dài
capture log close  // Đóng log cũ nếu còn mở

* --- 0.2 Mở log file để ghi lại toàn bộ phiên làm việc ---
log using "buoi1_stata_mau.log", replace text
display "=== Bắt đầu phiên làm việc: $S_DATE $S_TIME ==="

* --- 0.3 Thiết lập thư mục làm việc (SỬA đường dẫn cho máy của bạn) ---
* cd "C:/Users/tenban/Desktop/Stata_KinhTe"


/* ------------------------------------------------------------
   TẠO DỮ LIỆU MÔ PHỎNG
   63 tỉnh x 8 năm (2015–2022) = 504 quan sát
   ------------------------------------------------------------ */

* Tạo khung dữ liệu bảng (panel data)
set seed 2024          // Đặt seed để kết quả tái lặp được

* Số tỉnh và số năm
local n_prov = 63
local n_year = 8       // 2015 → 2022

* Tạo dataset rỗng với n_prov * n_year quan sát
set obs `= `n_prov' * `n_year''

* ---- Tạo biến province (mã tỉnh 1–63) và year (2015–2022) ----
gen province = ceil(_n / `n_year')
gen year      = 2014 + mod(_n - 1, `n_year') + 1

* ---- Tạo tên tỉnh (đại diện 63 tỉnh/thành) ----
label define prov_lbl  ///
     1 "Ha Noi"          2 "Vinh Phuc"       3 "Bac Ninh"       ///
     4 "Quang Ninh"      5 "Hai Duong"       6 "Hai Phong"      ///
     7 "Hung Yen"        8 "Thai Binh"       9 "Ha Nam"         ///
    10 "Nam Dinh"        11 "Ninh Binh"      12 "Thanh Hoa"     ///
    13 "Nghe An"         14 "Ha Tinh"        15 "Quang Binh"    ///
    16 "Quang Tri"       17 "Thua Thien Hue" 18 "Da Nang"       ///
    19 "Quang Nam"       20 "Quang Ngai"     21 "Binh Dinh"     ///
    22 "Phu Yen"         23 "Khanh Hoa"      24 "Ninh Thuan"    ///
    25 "Binh Thuan"      26 "Kon Tum"        27 "Gia Lai"       ///
    28 "Dak Lak"         29 "Dak Nong"       30 "Lam Dong"      ///
    31 "Binh Phuoc"      32 "Tay Ninh"       33 "Binh Duong"    ///
    34 "Dong Nai"        35 "Ba Ria VT"      36 "TP HCM"        ///
    37 "Long An"         38 "Tien Giang"     39 "Ben Tre"       ///
    40 "Tra Vinh"        41 "Vinh Long"      42 "Dong Thap"     ///
    43 "An Giang"        44 "Kien Giang"     45 "Can Tho"       ///
    46 "Hau Giang"       47 "Soc Trang"      48 "Bac Lieu"      ///
    49 "Ca Mau"          50 "Ha Giang"       51 "Cao Bang"      ///
    52 "Bac Kan"         53 "Tuyen Quang"    54 "Lao Cai"       ///
    55 "Yen Bai"         56 "Thai Nguyen"    57 "Lang Son"      ///
    58 "Bac Giang"       59 "Phu Tho"        60 "Dien Bien"     ///
    61 "Lai Chau"        62 "Son La"         63 "Hoa Binh"
label values province prov_lbl

* ---- Vùng kinh tế (6 vùng) ----
gen vung = .
replace vung = 1 if inrange(province, 50, 63)          // Trung du & MN phía Bắc
replace vung = 2 if inrange(province,  1, 11)          // Đồng bằng sông Hồng
replace vung = 3 if inrange(province, 12, 17)          // Bắc Trung Bộ
replace vung = 4 if inrange(province, 18, 25)          // Nam Trung Bộ
replace vung = 5 if inrange(province, 26, 30)          // Tây Nguyên
replace vung = 6 if inrange(province, 31, 49)          // Đông Nam Bộ & ĐBSCL

label define vung_lbl 1 "Trung du & MN phia Bac" 2 "DBSH"         ///
                      3 "Bac Trung Bo"            4 "Nam Trung Bo" ///
                      5 "Tay Nguyen"              6 "DN Bo & DBSCL"
label values vung vung_lbl
label variable vung "Vung kinh te (1-6)"

* ---- Khu vực (urban/rural dummy) ----
gen urban = (inlist(province, 1, 6, 18, 36, 45))   // 5 thành phố lớn
label variable urban "Do thi lon (1=Yes)"
label define urban_lbl 0 "Nong thon/TT nho" 1 "Thanh pho lon"
label values urban urban_lbl

* ---- GDP bình quân đầu người (triệu VND, giá 2015) ----
* Mô hình: GDP = base_prov + growth_trend + shock_nam + error
gen base_gdp = .
replace base_gdp = 80  + rnormal(0, 5)  if vung == 2   // ĐBSH (khá cao)
replace base_gdp = 200 + rnormal(0, 20) if urban == 1  // TP lớn (rất cao)
replace base_gdp = 40  + rnormal(0, 5)  if vung == 1   // TD&MN phía Bắc
replace base_gdp = 45  + rnormal(0, 5)  if vung == 3   // Bắc Trung Bộ
replace base_gdp = 60  + rnormal(0, 8)  if vung == 4   // Nam Trung Bộ
replace base_gdp = 55  + rnormal(0, 6)  if vung == 5   // Tây Nguyên
replace base_gdp = 90  + rnormal(0, 10) if vung == 6 & urban == 0  // ĐNB & ĐBSCL

gen gdp = base_gdp * (1 + 0.065)^(year - 2015) + rnormal(0, 3)
replace gdp = round(gdp, 0.1)
replace gdp = . if runiform() < 0.02   // Tạo ~2% missing (thực tế)
drop base_gdp
label variable gdp "GDP binh quan dau nguoi (trieu VND, gia 2015)"

* ---- FDI (triệu USD) ----
gen fdi = max(0, 50 + 20*urban + 15*(vung==2) + 10*(vung==6) ///
              + (year-2015)*8 + rnormal(0, 15))
replace fdi = round(fdi, 0.01)
replace fdi = -5 + rnormal(0,2) if runiform() < 0.03  // Một số âm (số liệu lỗi)
replace fdi = . if runiform() < 0.015
label variable fdi "Von FDI thuc hien (trieu USD)"

* ---- Thương mại / Xuất nhập khẩu (tỷ VND) ----
gen trade = max(0, 500 + 300*urban + 200*(vung==2) + ///
                (year-2015)*40 + rnormal(0, 80))
replace trade = round(trade, 1)
label variable trade "Kim ngach xuat nhap khau (ty VND)"

* ---- Dân số (nghìn người) ----
gen pop = 500 + (province - 1)*20 + 2000*urban + rnormal(0, 100)
replace pop = round(pop, 1)
replace pop = max(pop, 150)    // Tỉnh nhỏ nhất ~150 nghìn
label variable pop "Dan so (nghin nguoi)"

* ---- Tỷ lệ lao động qua đào tạo (%) ----
gen lao_dong_dt = 15 + 10*(vung==2) + 8*(vung==4) + 15*urban ///
                  + (year-2015)*1.2 + rnormal(0, 3)
replace lao_dong_dt = min(max(lao_dong_dt, 5), 85)
replace lao_dong_dt = round(lao_dong_dt, 0.1)
label variable lao_dong_dt "Ti le lao dong qua dao tao (%)"

* ---- Số doanh nghiệp hoạt động (doanh nghiệp) ----
gen doanh_nghiep = round(200 + 500*urban + 100*(vung==2) ///
                         + (year-2015)*30 + rnormal(0, 50), 1)
replace doanh_nghiep = max(doanh_nghiep, 50)
label variable doanh_nghiep "So doanh nghiep hoat dong"

* ---- Nhãn tổng quan dataset ----
label data "Du lieu panel 63 tinh/thanh VN, 2015-2022 (Mo phong)"
notes: Du lieu nay hoan toan mo phong phuc vu giang day Stata.
notes: Tao boi do-file buoi1_stata_mau.do – Seed = 2024

* Sắp xếp theo tỉnh và năm
sort province year

display "=== Da tao xong du lieu mo phong: `= _N' quan sat ==="


/* ============================================================
   PHẦN 1: LƯU VÀ NHẬP/XUẤT DỮ LIỆU
   ============================================================ */

display _newline(2) "===== PHẦN 1: LƯU VÀ NHẬP/XUẤT DỮ LIỆU ====="

* --- Lưu file Stata (.dta) – định dạng bản địa ---
save "panel_vietnam_mophong.dta", replace
display "✓ Da luu: panel_vietnam_mophong.dta"

* --- Xuất sang CSV để mở trong Excel ---
export delimited using "panel_vietnam_mophong.csv", replace
display "✓ Da xuat: panel_vietnam_mophong.csv"

* --- Thử nhập lại từ CSV (mô phỏng quy trình thực tế) ---
import delimited "panel_vietnam_mophong.csv", delimiter(",") clear
display "✓ Da nhap lai tu CSV"

* --- Quay lại dùng file .dta (nhanh hơn, giữ metadata) ---
use "panel_vietnam_mophong.dta", clear
display "✓ Da mo file .dta"

/*
   GHI NHỚ:
   - import excel  "file.xlsx", firstrow clear   → Nhập từ Excel
   - import delimited "file.csv", delimiter(",") clear → Nhập từ CSV
   - use "file.dta", clear                        → Mở file Stata
   - save "file.dta", replace                     → Lưu Stata
   - export excel using "file.xlsx", firstrow(variables) replace
*/


/* ============================================================
   PHẦN 2: KHÁM PHÁ DỮ LIỆU
   ============================================================ */

display _newline(2) "===== PHẦN 2: KHÁM PHÁ DỮ LIỆU ====="

* --- 2.1 Cấu trúc tổng quát ---
describe
/* Kết quả: tên biến, kiểu (float/int/str), nhãn biến, số obs */

describe gdp fdi trade pop vung urban
/* Mô tả chỉ một số biến cụ thể */

* --- 2.2 Thông tin chi tiết từng biến ---
codebook gdp
/* Hiển thị: loại biến, #unique, min, max, missing, phân vị */

codebook vung, tab(10)
/* tab(10): hiện tối đa 10 nhãn giá trị */

codebook province gdp fdi vung urban
/* Kiểm tra nhanh nhiều biến */

* --- 2.3 Xem dữ liệu dạng bảng ---
browse
/* Mở cửa sổ Data Browser – chỉ đọc, không sửa */

browse province year gdp fdi if year == 2022
/* Xem dữ liệu năm 2022 – chỉ một số biến */

* --- 2.4 Liệt kê quan sát ---
list province year gdp in 1/10
/* In 10 quan sát đầu tiên */

list province year gdp fdi if province == 1
/* In tất cả quan sát của Hà Nội (province=1) */

list if gdp < 0
/* Kiểm tra giá trị âm bất thường */

* --- 2.5 Kiểm tra giá trị thiếu (QUAN TRỌNG) ---
count if gdp == .
/* Đếm số obs có gdp bị thiếu */

count if fdi == .

misstable summarize
/* Tổng hợp missing toàn bộ dataset */

misstable patterns
/* Mẫu missing: biến nào thường missing cùng nhau */

display "==> LƯU Ý: Trong Stata, giá trị thiếu = dấu chấm (.)"
display "==> LUON kiểm tra missing trước khi phân tích!"


/* ============================================================
   PHẦN 3: THỐNG KÊ MÔ TẢ
   ============================================================ */

display _newline(2) "===== PHẦN 3: THỐNG KÊ MÔ TẢ ====="

* --- 3.1 summarize cơ bản ---
summarize gdp fdi trade pop lao_dong_dt
/* obs, mean, std.dev, min, max */

* --- 3.2 summarize chi tiết (có phân vị) ---
summarize gdp, detail
/* Thêm: p1, p5, p25, median, p75, p95, p99, skewness, kurtosis */

summarize fdi, detail
/* FDI thường phân phối lệch phải – kiểm tra skewness */

* --- 3.3 Bảng tần suất (tabulate) ---
tabulate vung
/* Số quan sát theo từng vùng kinh tế */

tabulate year
/* Số quan sát theo từng năm – kiểm tra bảng cân bằng */

tabulate vung urban
/* Bảng chéo 2 chiều: vùng x loại đô thị */

tabulate vung, summarize(gdp)
/* GDP trung bình theo từng vùng – kết hợp tab và sum */

* --- 3.4 Ma trận tương quan ---
correlate gdp fdi trade pop
/* Ma trận hệ số Pearson */

pwcorr gdp fdi trade lao_dong_dt, star(0.05)
/* Kèm dấu * nếu p < 0.05; ** nếu p < 0.01 */

* --- 3.5 Tính theo nhóm (bysort) ---
bysort vung: summarize gdp fdi
/* Thống kê mô tả GDP & FDI theo từng vùng */

bysort year: summarize gdp
/* Xu hướng GDP theo năm */

bysort urban: summarize gdp fdi lao_dong_dt
/* So sánh đô thị vs nông thôn */

* --- 3.6 Biểu đồ phân phối ---
histogram gdp, normal title("Phân phối GDP bình quân đầu người") ///
    xtitle("GDP (triệu VND)") ytitle("Mật độ") kdensity
/* normal: đường phân phối chuẩn; kdensity: ước lượng hạt nhân */

histogram fdi, normal title("Phân phối FDI") ///
    xtitle("FDI (triệu USD)") ytitle("Mật độ")
/* FDI thường lệch phải – thấy rõ trong histogram */

* Sau khi tạo ln_gdp ở phần 4, so sánh:
* kdensity ln_gdp, title("Phân phối ln(GDP)")


/* ============================================================
   PHẦN 4: TẠO VÀ BIẾN ĐỔI BIẾN
   ============================================================ */

display _newline(2) "===== PHẦN 4: TẠO VÀ BIẾN ĐỔI BIẾN ====="

* --- 4.1 Tạo biến logarithm (log) ---
gen ln_gdp = ln(gdp)
label variable ln_gdp "Log GDP binh quan dau nguoi"

gen ln_fdi = ln(fdi + 1)      // +1 để tránh ln(0) khi FDI=0
label variable ln_fdi "Log(FDI + 1)"

gen ln_pop = ln(pop)
label variable ln_pop "Log dan so"

* Kiểm tra ngay sau khi tạo
summarize ln_gdp ln_fdi ln_pop
histogram ln_gdp, normal title("Phân phối ln(GDP)") xtitle("ln(GDP)")
/* So sánh với histogram gdp ở trên – phân phối chuẩn hơn */

* --- 4.2 Tạo biến bình phương (kiểm định phi tuyến) ---
gen gdp_sq = gdp^2
label variable gdp_sq "GDP binh phuong"

* --- 4.3 Tốc độ tăng trưởng GDP (cần xtset trước) ---
* Thiết lập cấu trúc panel: tỉnh là đơn vị chéo, năm là thời gian
xtset province year

gen growth_gdp = (gdp - L.gdp) / L.gdp * 100
/* L.gdp = lag 1 năm của gdp */
label variable growth_gdp "Toc do tang truong GDP (%/nam)"

* Kiểm tra
summarize growth_gdp
list province year gdp growth_gdp if province == 1
/* 2015 sẽ có growth = . vì không có L.gdp */

* --- 4.4 replace – thay thế giá trị theo điều kiện ---
* Sửa FDI âm (lỗi số liệu) → 0
count if fdi < 0
replace fdi = 0 if fdi < 0
/* Kiểm tra lại */
count if fdi < 0      // Phải bằng 0

* Đánh dấu outlier GDP
gen gdp_outlier = (gdp > 300) if gdp != .
label variable gdp_outlier "GDP outlier (>300 trieu VND)"
tab gdp_outlier

* --- 4.5 Tạo biến giả (dummy) từ biến liên tục ---
* Phân nhóm GDP thành 3 mức
gen gdp_nhom = .
replace gdp_nhom = 1 if gdp < 50
replace gdp_nhom = 2 if gdp >= 50  & gdp < 100
replace gdp_nhom = 3 if gdp >= 100 & gdp != .

label define gdp_nhom_lbl 1 "Thap (<50)" 2 "Trung binh (50-100)" ///
                           3 "Cao (>=100)"
label values gdp_nhom gdp_nhom_lbl
label variable gdp_nhom "Nhom GDP (1=Thap, 2=TB, 3=Cao)"

tab gdp_nhom

* Cách thay thế bằng recode (ngắn gọn hơn)
recode gdp (0/49.99 = 1 "Thap") (50/99.99 = 2 "Trung binh") ///
           (100/max  = 3 "Cao"), gen(gdp_nhom2)
label variable gdp_nhom2 "Nhom GDP – recode (kiem tra)"
tab gdp_nhom gdp_nhom2     // Hai biến phải giống nhau

* --- 4.6 Nhãn biến tiếng Việt ---
label variable gdp          "GDP binh quan dau nguoi (trieu VND)"
label variable fdi          "Von FDI thuc hien (trieu USD)"
label variable trade        "Kim ngach XNK (ty VND)"
label variable pop          "Dan so (nghin nguoi)"
label variable lao_dong_dt  "Ti le lao dong qua dao tao (%)"
label variable doanh_nghiep "So doanh nghiep hoat dong"

* --- 4.7 Kiểm tra cuối phần 4 ---
describe
summarize gdp ln_gdp growth_gdp gdp_nhom fdi ln_fdi


/* ============================================================
   PHẦN 5: GHÉP DỮ LIỆU – MERGE VÀ APPEND
   ============================================================ */

display _newline(2) "===== PHẦN 5: GHÉP DỮ LIỆU ====="

* Lưu dataset chính trước khi demo ghép
save "panel_main.dta", replace

/* ----- 5A. DEMO merge 1:1 -----
   Tách dataset thành 2 file rồi ghép lại để minh hoạ */

* File A: Chỉ giữ GDP, dân số, vùng
use "panel_main.dta", clear
keep province year gdp pop vung urban ln_gdp
save "file_A_gdp.dta", replace
display "✓ Da tao file_A_gdp.dta"

* File B: Chỉ giữ FDI, thương mại, lao động
use "panel_main.dta", clear
keep province year fdi trade lao_dong_dt doanh_nghiep
save "file_B_fdi.dta", replace
display "✓ Da tao file_B_fdi.dta"

* --- Ghép merge 1:1 theo khóa (province, year) ---
use "file_A_gdp.dta", clear
merge 1:1 province year using "file_B_fdi.dta"

display _newline "==> Kiểm tra biến _merge sau khi ghép:"
tab _merge
/*
   _merge == 1: quan sát chỉ có trong master (file A)
   _merge == 2: quan sát chỉ có trong using  (file B)
   _merge == 3: matched – khớp cả hai file   ← mong muốn
*/

* Giữ chỉ những quan sát khớp
keep if _merge == 3
drop _merge
display "✓ Sau merge 1:1: `= _N' quan sat"

/* ----- 5B. DEMO merge m:1 -----
   Thêm đặc trưng cấp tỉnh (không đổi theo năm) vào panel */

* Tạo file đặc trưng tỉnh (province-level characteristics)
use "panel_main.dta", clear
keep province vung urban          // Giữ biến không đổi theo năm
bysort province: keep if _n == 1  // Chỉ giữ 1 dòng mỗi tỉnh
save "prov_chars.dta", replace
display "✓ Da tao prov_chars.dta (`= _N' tinh)"

* Dataset panel (nhiều dòng mỗi tỉnh)
use "panel_main.dta", clear
drop vung urban      // Xóa để demo ghép lại
merge m:1 province using "prov_chars.dta"
tab _merge
drop _merge
display "✓ Sau merge m:1: da them lai bien vung va urban"

/* ----- 5C. DEMO append -----
   Chia dữ liệu thành 2 giai đoạn rồi nối lại */

use "panel_main.dta", clear
keep if year <= 2018
save "data_2015_2018.dta", replace
display "✓ Da tao data_2015_2018.dta"

use "panel_main.dta", clear
keep if year >= 2019
save "data_2019_2022.dta", replace
display "✓ Da tao data_2019_2022.dta"

* Nối hai file lại
use "data_2015_2018.dta", clear
append using "data_2019_2022.dta"
sort province year
display "✓ Sau append: `= _N' quan sat"

* Kiểm tra sau append
duplicates report province year
/* Không được có duplicate – nếu có thì lỗi overlap */

tab year         // Phải có đủ 8 năm
tabstat gdp, by(year) stat(n mean)


/* ============================================================
   PHẦN 6: TỔNG KẾT VÀ LƯU KẾT QUẢ
   ============================================================ */

display _newline(2) "===== PHẦN 6: TỔNG KẾT ====="

* Mở dataset chính (đầy đủ)
use "panel_main.dta", clear
xtset province year

* --- Thống kê mô tả cuối cùng (kiểm tra toàn bộ) ---
describe
codebook gdp fdi trade pop vung urban
summarize gdp fdi trade pop lao_dong_dt doanh_nghiep
pwcorr gdp fdi trade lao_dong_dt, star(0.05)
bysort vung: summarize gdp fdi

* --- Lưu dataset sạch cuối cùng ---
save "panel_clean_buoi1.dta", replace
display "✓ Da luu dataset sach: panel_clean_buoi1.dta"
display "  Dataset san sang cho phan tich Buoi 2 (OLS, FE, RE, IV...)"

* --- Đóng log ---
display _newline "=== Ket thuc phien lam viec: $S_DATE $S_TIME ==="
log close

/* ============================================================
   GHI CHÚ CHO HỌC VIÊN
   ============================================================
   
   FILE ĐÃ TẠO SAU KHI CHẠY DO-FILE NÀY:
   ✓ panel_vietnam_mophong.dta  – Dataset gốc (63 tỉnh, 2015–2022)
   ✓ panel_vietnam_mophong.csv  – Bản CSV để mở bằng Excel
   ✓ panel_main.dta             – Dataset chính (đầy đủ biến)
   ✓ panel_clean_buoi1.dta      – Dataset sạch cho Buổi 2
   ✓ file_A_gdp.dta             – File A (demo merge)
   ✓ file_B_fdi.dta             – File B (demo merge)
   ✓ prov_chars.dta             – Đặc trưng tỉnh (demo merge m:1)
   ✓ data_2015_2018.dta         – Giai đoạn 1 (demo append)
   ✓ data_2019_2022.dta         – Giai đoạn 2 (demo append)
   ✓ buoi1_stata_mau.log        – Log toàn bộ phiên làm việc
   
   LỆNH VIẾT TẮT THƯỜNG DÙNG:
   summarize → sum       generate → gen
   regress   → reg       tabulate → tab
   describe  → des       label    → la
   
   PHÍM TẮT DO-FILE EDITOR:
   Ctrl+D          → Chạy toàn bộ file
   Ctrl+Shift+D    → Chạy phần được chọn
   Ctrl+/          → Comment/Uncomment dòng
   
   ============================================================ */

* ===== KẾT THÚC DO-FILE BUỔI 1 =====
