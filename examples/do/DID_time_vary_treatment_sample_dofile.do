// 1. CÀI ĐẶT GÓI CẦN THIẾT (Chỉ chạy 1 lần)
// CSDID phụ thuộc vào gói DRDID, hãy chắc chắn bạn đã cài cả hai
* ssc install drdid, replace
* ssc install csdid, replace
* ssc install reghdfe, replace  // Dùng để so sánh

// 2. KHỞI TẠO DỮ LIỆU GIẢ LẬP
clear all
set seed 12345  // Đặt hạt giống để kết quả lặp lại được
set obs 1000    // Tạo 1000 đơn vị (ví dụ: tỉnh/người)

gen id = _n     // Tạo ID cho từng đơn vị

// Tạo cấu trúc Panel: 1000 đơn vị x 15 năm (2000 - 2014)
expand 15
bysort id: gen year = _n + 1999

// --- 3. PHÂN CHIA NHÓM CAN THIỆP (COHORTS) ---
// Chúng ta sẽ có 3 nhóm:
// Nhóm Never Treated (Không bao giờ bị): gvar = 0
// Nhóm Can thiệp sớm (Early cohort - 2006): gvar = 2006
// Nhóm Can thiệp muộn (Late cohort - 2010): gvar = 2010

gen gvar = 0  // Mặc định là nhóm đối chứng (Never treated)

// Gán 300 ông vào nhóm can thiệp 2006
replace gvar = 2006 if id <= 300 

// Gán 300 ông vào nhóm can thiệp 2010
replace gvar = 2010 if id > 300 & id <= 600

// Các ông còn lại (id > 600) sẽ giữ gvar = 0 (Never Treated)

// --- 4. TẠO BIẾN TRẠNG THÁI (D) ---
// D = 1 nếu năm hiện tại >= năm được can thiệp
gen treated = 0
replace treated = 1 if (year >= gvar) & (gvar != 0)

// --- 5. TẠO BIẾN KẾT QUẢ (Y) VỚI HIỆU ỨNG ĐỘNG (QUAN TRỌNG) ---
// Y = Unit FE + Time FE + Treatment Effect + Error
gen unit_fe = rnormal(0, 1)    // Đặc tính riêng cố định
gen time_fe = rnormal(0, 0.5)  // Cú sốc thời gian chung

// Tạo Treatment Effect (TE)
// Giả định: Tác động tăng dần theo thời gian.
// Năm đầu tiên = 2, Năm thứ 2 = 3, Năm thứ 3 = 4...
gen te = 0
replace te = (year - gvar + 1) + 1 if treated == 1

// Tính Y cuối cùng
gen y = unit_fe + time_fe + te + rnormal(0, 1)

// =========================================================
// PHẦN PHÂN TÍCH
// =========================================================

// --- CÁCH 1: TWFE CŨ (Hồi quy Fixed Effects truyền thống) ---
// Đây là cách "máy xay sinh tố", ra 1 con số trung bình chung chung
reghdfe y treated, absorb(id year) cluster(id)

// Nhận xét: Bạn sẽ thấy hệ số loanh quanh khoảng trung bình của các năm.
// Nhưng nó che giấu sự thật là tác động đang tăng dần.

// --- CÁCH 2: CSDID (Callaway & Sant'Anna) ---

/* Cú pháp:
   ivar: Biến định danh (id)
   time: Biến thời gian (year)
   gvar: Biến xác định nhóm (Quan trọng: phải chứa NĂM can thiệp, hoặc 0 nếu không can thiệp)
*/

csdid y, ivar(id) time(year) gvar(gvar)

// 1. Tác động tổng thể (Simple Average)
// Nó cố gắng gộp lại thành 1 số như TWFE, nhưng với trọng số chuẩn xác hơn
estat simple 

// 2. Tác động theo nhóm (Group-Time Effects)
// "Những viên gạch Lego" rời rạc mà ta đã thảo luận. 
// Xem ông nhóm 2006 bị tác động thế nào vào năm 2007, 2008...
estat group

// 3. Event Study (Quan trọng nhất)
// Trả lời câu hỏi: Tác động thay đổi thế nào theo thời gian (dynamic)?
estat event

// --- VẼ ĐỒ THỊ ---
// CSDID tích hợp sẵn lệnh vẽ đồ thị rất đẹp để xem xu hướng song song và dynamic effect
csdid_plot, title("Dynamic Treatment Effect using CSDID")