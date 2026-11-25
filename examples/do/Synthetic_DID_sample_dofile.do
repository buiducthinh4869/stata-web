// 1. CÀI ĐẶT GÓI CẦN THIẾT
// Bạn cần gói sdid và gói graph (đôi khi cần cài thêm)
* ssc install sdid, replace
* ssc install drdid, replace // Một số dependency
* ssc install reghdfe, replace

// 2. TẠO DỮ LIỆU GIẢ LẬP (Data Generation)
clear all
set seed 9999
set obs 50       // 50 Tỉnh thành
gen id = _n
expand 30        // 30 Năm (Quan sát dài)
bysort id: gen t = _n

// --- CẤU TẠO CÁC YẾU TỐ ẨN (Latent Factors) ---
// Giả sử có một "Cú sốc vĩ mô" (Market Trend) hình sin phức tạp
gen common_shock = sin(t/4) + (t/10)^2 

// --- PHÂN LOẠI ĐƠN VỊ ---
// ID 1-5: Nhóm CAN THIỆP (Treated Group). Chịu ảnh hưởng rất mạnh bởi common_shock
gen group_type = "Treated" if id <= 5

// ID 6-20: Nhóm ĐỐI CHỨNG TỐT (Good Controls). Cũng chịu ảnh hưởng giống Treated.
replace group_type = "Good Control" if id > 5 & id <= 20

// ID 21-50: Nhóm ĐỐI CHỨNG XẤU (Bad Controls). Xu hướng hoàn toàn khác (ví dụ: đi ngang hoặc nhiễu).
replace group_type = "Bad Control" if id > 20

// --- TẠO BIẾN Y (KẾT QUẢ) ---
gen y = 0
// Tạo mức nền (Unit FE) + Thời gian (Time FE)
replace y = id/5 + t/10 

// Thêm yếu tố nhiễu loạn:
// - Nhóm Treated và Good Control: Đi theo hình Sin (nhịp điệu giống nhau -> Song song tốt)
// - Nhóm Bad Control: Không đi theo hình Sin (làm hỏng Parallel Trends)
replace y = y + 3*common_shock if group_type == "Treated" | group_type == "Good Control"
replace y = y + rnormal(0, 1) // Thêm sai số ngẫu nhiên

// --- TẠO TÁC ĐỘNG THỰC (TREATMENT EFFECT) ---
// Giả sử can thiệp xảy ra từ năm thứ 21 (t >= 21) cho nhóm Treated (id <= 5)
gen D = 0
replace D = 1 if id <= 5 & t >= 21

// Gán tác động thật (ATT) = +10 điểm
replace y = y + 10 if D == 1

// --- MINH HỌA DỮ LIỆU ---
// Để bạn thấy sự khác biệt trước khi chạy mô hình
twoway (line y t if id==1, lcolor(red) lwidth(thick)) || ///
       (line y t if id==6, lcolor(blue) lpattern(dash)) || ///
       (line y t if id==30, lcolor(gs12)), ///
       legend(label(1 "Treated Unit") label(2 "Good Control") label(3 "Bad Control")) ///
       title("Bối cảnh: Xu hướng lộn xộn") subtitle("Treated giống Good Control, khác Bad Control") ///
       xline(21)
	   
// =========================================================
// 3. SO SÁNH PHÂN TÍCH
// =========================================================

// --- CÁCH 1: DiD TRUYỀN THỐNG (TWFE) ---
// DiD lấy trung bình tất cả (cả tốt lẫn xấu). Nhóm xấu làm lệch xu hướng song song.
reghdfe y D, absorb(id t) cluster(id)
// KẾT QUẢ: Hệ số thường sẽ KHÔNG bằng 10 (có thể bị chệch lớn hoặc SE rất to).

// --- CÁCH 2: SDID (Synthetic Difference-in-Differences) ---
// SDID sẽ tự động phát hiện nhóm Good Control và Bad Control để gán trọng số
// vce(bootstrap) là chuẩn mực cho SDID để tính sai số chuẩn
sdid y id t D, vce(bootstrap) seed(123) reps(50) graph 

/* 
   LƯU Ý QUAN TRỌNG VỀ ĐỒ THỊ CỦA SDID:
   Khi chạy xong lệnh trên, SDID sẽ xuất ra một đồ thị ghép rất đẹp.
   Bạn hãy nhìn vào đồ thị và đọc giải thích ở dưới của tôi.
*/