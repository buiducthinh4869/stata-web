# Stata Web

Mô phỏng Stata chạy hoàn toàn trong trình duyệt (Python/Pyodide), phục vụ giảng dạy các phương pháp suy diễn nhân quả: RCT, IV, DID (TWFE, didregress, csdid, sdid), PSM (psmatch2, teffects, kmatch), RDD (rdrobust, rdplot, rddensity), SCM (synth2).
Ngoài ra có các mô hình phi tuyến poisson, nbreg, ppmlhdfe, tobit, ologit/oprobit, mlogit và các lệnh dữ liệu decode, split, tsfill, fillin, contract, winsor2.

**Chuỗi thời gian:** dfuller, pperron, kpss, zandrews, arima, arch/garch/egarch/tarch, var, svar, irf, vecrank, vec, ardl (bounds test PSS/Kripfganz–Schneider).
**Dữ liệu bảng nâng cao:** xtabond2, xtabond, xtdpdsys (GMM, Hansen/Sargan, AR(1)/AR(2), Windmeijer), xtunitroot (llc, ips, ht, breitung, fisher, hadri), xtgls, xtpcse, xtscc, xttest0, xttest3, xtserial, xtcsd.

**Dùng ngay:** https://buiducthinh4869.github.io/stata-web/

- Mở một thư mục trên máy (Chrome/Edge) để `use`, `save`, `graph export`, `esttab using`, `outreg2`, `log using` đọc/ghi trực tiếp.
- Trình duyệt khác: kéo thả file lên, tải kết quả về (từng file hoặc .zip).
- Dữ liệu không rời khỏi máy người dùng.

## Giới hạn
- Hỗ trợ khoảng 210 lệnh có trong bộ do-file mẫu (thư mục `examples/do`), không phải toàn bộ Stata.
- Bộ sinh số ngẫu nhiên khác Stata: dữ liệu mô phỏng bằng `rnormal`/`runiform` cho kết quả gần nhưng không trùng từng chữ số.
- `teffects psmatch`: sai số chuẩn theo Abadie–Imbens (2006), chưa hiệu chỉnh cho propensity score ước lượng (AI 2016).
- `kmatch`: băng thông theo quy tắc pair-matching, sai số chuẩn coi trọng số là cố định.
- `ppmlhdfe`: chưa hỗ trợ `d()`/`predict`; `margins` chưa hỗ trợ sau poisson, nbreg, tobit, ologit, mlogit.
- `synth2`: tối ưu hóa V bằng Nelder–Mead; trọng số có thể lệch nhẹ so với Stata.

- `xtabond2` với difference GMM in ra Wald chi2 đúng thay vì `chi2(0) = .` như bản gốc; `xtdpdsys` one-step không robust có thể lệch nhẹ sai số chuẩn so với `xtdpd`.
- `xtunitroot breitung, robust` chưa hỗ trợ; `xtcointtest` chưa có.

## Phát triển
- Engine: `py/stata/` (thông dịch do-file, quản lý dữ liệu, ước lượng, đồ thị). `py/rdpkg/`: phần tính toán của gói rdrobust/rddensity (GPL-3.0, Calonico, Cattaneo, Farrell, Titiunik; Cattaneo, Jansson, Ma).
- Chạy thử một do-file trên máy: `python3 tests/run_do.py examples/do/RDD_sample_dofile.do /tmp/work`
- Sau khi sửa engine: `python3 tools_build.py` để tạo lại `py.zip`.
