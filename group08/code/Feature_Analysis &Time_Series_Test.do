import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\AUDUSD_D1.csv"
* 数据格式清洗
gen date = date(time, "YMDhms")  // 将字符型时间转为 Stata 日期型
format date %td  // 日期格式化为 "YYYY-MM-DD"
drop if date < td(12dec2017)
drop if missing(close, date)  // 删除收盘价或日期为空的观测
sort date  // 按日期排序（确保时序正确）

* 只保留核心变量（日期+收盘价），简化数据
keep date close
order date close

tsset date
/* ===================== 2. 计算关键指标 ===================== */
* 1. 对数收益率（日度）
gen log_return = ln(close) - ln(L.close)  // L. 表示滞后1期
label var log_return "日度对数收益率"

* 2. 历史波动率（20日滚动标准差，年化）
gen vol_daily = .
label var vol_daily "20日滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    local end   = `i'
    summarize log_return in `start'/`end'
    replace vol_daily = r(sd) in `i'
}

* 年化波动率
gen vol_annual = vol_daily * sqrt(252)
label var vol_annual "20日年化波动率"
* 删除指标计算产生的空值（前19期无波动率数据）
drop if missing(log_return, vol_annual)


/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
* -------------------- 图1：收盘价时序图 --------------------
twoway line close date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("AUDUSD Daily Closing Price", size(medium)) ///
    ytitle("Closing Price", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/AUDUSD_close.png", replace dpi(300)
* -------------------- 图2：对数收益率时序图 --------------------
twoway line log_return date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("AUDUSD Daily Logarithmic Return", size(medium)) ///
    ytitle("Logarithmic Return", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.2f) angle(0)) /// y轴标签保留2位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/AUDUSD_log_return.png", replace dpi(300)
* -------------------- 图3：历史波动率时序图 --------------------
//法1（成功）
twoway line vol_annual date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("AUDUSD 20-Day Rolling Historical Volatility", size(medium)) ///
    ytitle("Annualized Volatility", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/AUDUSD_volatility.png", replace dpi(300)
* --------------------3张图放一起 --------------------
twoway line close date, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return date, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual date, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Day Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)

* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("AUDUSD Daily Timing Analysis")

* 5）保存最终图片
graph export "three_combined.png", replace dpi(300)

* -------------- 1. 平稳性检验（ADF检验）--------------
* 原假设：存在单位根（不平稳）
* p值 < 0.05 → 拒绝原假设 → 平稳
dfuller log_return, lags(3) // 滞后3期，通用金融数据



* -------------- 2. 自相关检验（ACF PACF）--------------
* 查看前20期自相关与偏自相关
ac log_return, lags(20) title("AUDUSD Autocorrelation Function")
pac log_return, lags(20) title("AUDUSD Partial Autocorrelation Function")


* -------------- 3. Ljung-Box 检验（Q检验）--------------
* 原假设：序列无自相关
corrgram log_return,lags(20)
* p值 < 0.05 → 存在自相关
wntestq log_return, lags(10)   // 滞后10期
wntestq log_return, lags(20)   // 滞后20期


* -------------- 4. ARCH效应检验（异方差检验）--------------
* 法2直接用官方命令检验（最简单）
reg log_return
estat archlm
* -------------- 分析BTC---
import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\BTCUSD_D1.csv"
gen date = date(time, "YMDhms")  // 将字符型时间转为 Stata 日期型
format date %td  // 日期格式化为 "YYYY-MM-DD"
drop if date < td(12dec2017)
drop if missing(close, date)  // 删除收盘价或日期为空的观测
sort date  // 按日期排序（确保时序正确）

* 只保留核心变量（日期+收盘价），简化数据
keep date close
order date close

tsset date
/* ===================== 2. 计算关键指标 ===================== */
* 1. 对数收益率（日度）
gen log_return = ln(close) - ln(L.close)  // L. 表示滞后1期
label var log_return "日度对数收益率"

* 2. 历史波动率（20日滚动标准差，年化）
gen vol_daily = .
label var vol_daily "20日滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    local end   = `i'
    summarize log_return in `start'/`end'
    replace vol_daily = r(sd) in `i'
}

* 年化波动率
gen vol_annual = vol_daily * sqrt(252)
label var vol_annual "20日年化波动率"
* 删除指标计算产生的空值（前19期无波动率数据）
drop if missing(log_return, vol_annual)

/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
* -------------------- 图1：收盘价时序图 --------------------
twoway line close date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("BTCUSD Daily Closing Price", size(medium)) ///
    ytitle("Closing Price", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/BTCUSD_close.png", replace dpi(300)
* -------------------- 图2：对数收益率时序图 --------------------
twoway line log_return date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("BTCUSD Daily Logarithmic Return", size(medium)) ///
    ytitle("Logarithmic Return", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.2f) angle(0)) /// y轴标签保留2位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/BTCUSD_log_return.png", replace dpi(300)
* -------------------- 图3：历史波动率时序图 --------------------
//法1（成功）
twoway line vol_annual date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("BTCUSD 20-Day Rolling Historical Volatility", size(medium)) ///
    ytitle("Annualized Volatility", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/BTCUSD_volatility.png", replace dpi(300)
* --------------------3张图放一起 --------------------
twoway line close date, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return date, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual date, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Day Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)

* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("BTCUSD Daily Timing Analysis")

* 5）保存最终图片
graph export "three_combined.png", replace dpi(300)

* -------------- 1. 平稳性检验（ADF检验）--------------
* 原假设：存在单位根（不平稳）
* p值 < 0.05 → 拒绝原假设 → 平稳
dfuller log_return, lags(10) // 滞后10期，通用金融数据



* -------------- 2. 自相关检验（ACF PACF）--------------
* 查看前20期自相关与偏自相关
ac log_return, lags(20) title("BTCUSD Autocorrelation Function")
pac log_return, lags(20) title("BTCUSD Partial Autocorrelation Function")


* -------------- 3. Ljung-Box 检验（Q检验）--------------
* 原假设：序列无自相关
corrgram log_return,lags(20)
* p值 < 0.05 → 存在自相关
wntestq log_return, lags(10)   // 滞后10期
wntestq log_return, lags(20)   // 滞后20期


* -------------- 4. ARCH效应检验（异方差检验）--------------
* 法2直接用官方命令检验（最简单）
reg log_return
estat archlm

* -------------- 4. ETH分析--------------
import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\ETHUSD_D1.csv"
gen date = date(time, "YMDhms")  // 将字符型时间转为 Stata 日期型
format date %td  // 日期格式化为 "YYYY-MM-DD"
drop if date < td(12dec2017)
drop if missing(close, date)  // 删除收盘价或日期为空的观测
sort date  // 按日期排序（确保时序正确）

* 只保留核心变量（日期+收盘价），简化数据
keep date close
order date close

tsset date
/* ===================== 2. 计算关键指标 ===================== */
* 1. 对数收益率（日度）
gen log_return = ln(close) - ln(L.close)  // L. 表示滞后1期
label var log_return "日度对数收益率"

* 2. 历史波动率（20日滚动标准差，年化）
gen vol_daily = .
label var vol_daily "20日滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    local end   = `i'
    summarize log_return in `start'/`end'
    replace vol_daily = r(sd) in `i'
}

* 年化波动率
gen vol_annual = vol_daily * sqrt(252)
label var vol_annual "20日年化波动率"
* 删除指标计算产生的空值（前19期无波动率数据）
drop if missing(log_return, vol_annual)

/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
* -------------------- 图1：收盘价时序图 --------------------
twoway line close date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("ETHUSD Daily Closing Price", size(medium)) ///
    ytitle("Closing Price", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/ETHUSD_close.png", replace dpi(300)
* -------------------- 图2：对数收益率时序图 --------------------
twoway line log_return date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("ETHUSD Daily Logarithmic Return", size(medium)) ///
    ytitle("Logarithmic Return", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.2f) angle(0)) /// y轴标签保留2位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/ETHUSD_log_return.png", replace dpi(300)
* -------------------- 图3：历史波动率时序图 --------------------
//法1（成功）
twoway line vol_annual date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("ETHUSD 20-Day Rolling Historical Volatility", size(medium)) ///
    ytitle("Annualized Volatility", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/ETHUSD_volatility.png", replace dpi(300)
* --------------------3张图放一起 --------------------
twoway line close date, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return date, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual date, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Day Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)

* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("ETHUSD Daily Timing Analysis")

* 5）保存最终图片
graph export "three_combined.png", replace dpi(300)
destring A B C,replace
* -------------- 1. 平稳性检验（ADF检验）--------------
* 原假设：存在单位根（不平稳）
* p值 < 0.05 → 拒绝原假设 → 平稳
dfuller log_return, lags(10) // 滞后10期，通用金融数据



* -------------- 2. 自相关检验（ACF PACF）--------------
* 查看前20期自相关与偏自相关
ac log_return, lags(20) title("ETHUSD Autocorrelation Function")
pac log_return, lags(20) title("ETHUSD Partial Autocorrelation Function")


* -------------- 3. Ljung-Box 检验（Q检验）--------------
* 原假设：序列无自相关
corrgram log_return,lags(20)
* p值 < 0.05 → 存在自相关
wntestq log_return, lags(10)   // 滞后10期
wntestq log_return, lags(20)   // 滞后20期


* -------------- 4. ARCH效应检验（异方差检验）--------------
* 法2直接用官方命令检验（最简单）
reg log_return
estat archlm

* -------------- EUR分析--------------
import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\EURUSD_D1.csv"
gen date = date(time, "YMDhms")  // 将字符型时间转为 Stata 日期型
format date %td  // 日期格式化为 "YYYY-MM-DD"
drop if date < td(12dec2017)
drop if missing(close, date)  // 删除收盘价或日期为空的观测
sort date  // 按日期排序（确保时序正确）

* 只保留核心变量（日期+收盘价），简化数据
keep date close
order date close

tsset date
/* ===================== 2. 计算关键指标 ===================== */
* 1. 对数收益率（日度）
gen log_return = ln(close) - ln(L.close)  // L. 表示滞后1期
label var log_return "日度对数收益率"

* 2. 历史波动率（20日滚动标准差，年化）
gen vol_daily = .
label var vol_daily "20日滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    local end   = `i'
    summarize log_return in `start'/`end'
    replace vol_daily = r(sd) in `i'
}

* 年化波动率
gen vol_annual = vol_daily * sqrt(252)
label var vol_annual "20日年化波动率"
* 删除指标计算产生的空值（前19期无波动率数据）
drop if missing(log_return, vol_annual)

/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
* -------------------- 图1：收盘价时序图 --------------------
twoway line close date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("EURUSD Daily Closing Price", size(medium)) ///
    ytitle("Closing Price", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/EURUSD_close.png", replace dpi(300)
* -------------------- 图2：对数收益率时序图 --------------------
twoway line log_return date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("EURUSD Daily Logarithmic Return", size(medium)) ///
    ytitle("Logarithmic Return", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.2f) angle(0)) /// y轴标签保留2位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/EURUSD_log_return.png", replace dpi(300)
* -------------------- 图3：历史波动率时序图 --------------------
//法1（成功）
twoway line vol_annual date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("EURUSD 20-Day Rolling Historical Volatility", size(medium)) ///
    ytitle("Annualized Volatility", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/EURUSD_volatility.png", replace dpi(300)
* --------------------3张图放一起 --------------------
twoway line close date, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return date, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual date, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Day Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)

* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("EURUSD Daily Timing Analysis")

* 5）保存最终图片
graph export "three_combined.png", replace dpi(300)

* -------------- 1. 平稳性检验（ADF检验）--------------
* 原假设：存在单位根（不平稳）
* p值 < 0.05 → 拒绝原假设 → 平稳
dfuller log_return, lags(3) // 滞后10期，通用金融数据



* -------------- 2. 自相关检验（ACF PACF）--------------
* 查看前20期自相关与偏自相关
ac log_return, lags(20) title("EURUSD Autocorrelation Function")
pac log_return, lags(20) title("EURUSD Partial Autocorrelation Function")


* -------------- 3. Ljung-Box 检验（Q检验）--------------
* 原假设：序列无自相关
corrgram log_return,lags(20)
* p值 < 0.05 → 存在自相关
wntestq log_return, lags(10)   // 滞后10期
wntestq log_return, lags(20)   // 滞后20期


* -------------- 4. ARCH效应检验（异方差检验）--------------
* 法2直接用官方命令检验（最简单）
reg log_return
estat archlm

* -------------- GBP分析--------------
import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\GBPUSD_D1.csv"
gen date = date(time, "YMDhms")  // 将字符型时间转为 Stata 日期型
format date %td  // 日期格式化为 "YYYY-MM-DD"
drop if date < td(12dec2017)
drop if missing(close, date)  // 删除收盘价或日期为空的观测
sort date  // 按日期排序（确保时序正确）

* 只保留核心变量（日期+收盘价），简化数据
keep date close
order date close

tsset date
/* ===================== 2. 计算关键指标 ===================== */
* 1. 对数收益率（日度）
gen log_return = ln(close) - ln(L.close)  // L. 表示滞后1期
label var log_return "日度对数收益率"

* 2. 历史波动率（20日滚动标准差，年化）
gen vol_daily = .
label var vol_daily "20日滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    local end   = `i'
    summarize log_return in `start'/`end'
    replace vol_daily = r(sd) in `i'
}

* 年化波动率
gen vol_annual = vol_daily * sqrt(252)
label var vol_annual "20日年化波动率"
* 删除指标计算产生的空值（前19期无波动率数据）
drop if missing(log_return, vol_annual)

/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
* -------------------- 图1：收盘价时序图 --------------------
twoway line close date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("GBPUSD Daily Closing Price", size(medium)) ///
    ytitle("Closing Price", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/GBPUSD_close.png", replace dpi(300)
* -------------------- 图2：对数收益率时序图 --------------------
twoway line log_return date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("GBPUSD Daily Logarithmic Return", size(medium)) ///
    ytitle("Logarithmic Return", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.2f) angle(0)) /// y轴标签保留2位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/GBPUSD_log_return.png", replace dpi(300)
* -------------------- 图3：历史波动率时序图 --------------------
//法1（成功）
twoway line vol_annual date, ///
    lcolor("green") lwidth(medium) /// 绿色线条
    title("GBPUSD 20-Day Rolling Historical Volatility", size(medium)) ///
    ytitle("Annualized Volatility", size(small)) ///
    xtitle("Date", size(small)) ///
    ylabel(, format(%9.1f) angle(0)) /// y轴标签保留1位小数
    xlabel(#10, angle(45)) ///
    legend(off)
graph export "/mnt/GBPUSD_volatility.png", replace dpi(300)
* --------------------3张图放一起 --------------------
twoway line close date, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return date, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual date, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Day Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)

* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("GBPUSD Daily Timing Analysis")

* 5）保存最终图片
graph export "three_combined.png", replace dpi(300)

* -------------- 1. 平稳性检验（ADF检验）--------------
* 原假设：存在单位根（不平稳）
* p值 < 0.05 → 拒绝原假设 → 平稳
dfuller log_return, lags(3) // 滞后10期，通用金融数据



* -------------- 2. 自相关检验（ACF PACF）--------------
* 查看前20期自相关与偏自相关
ac log_return, lags(20) title("GBPUSD Autocorrelation Function")
pac log_return, lags(20) title("GBPUSD Partial Autocorrelation Function")


* -------------- 3. Ljung-Box 检验（Q检验）--------------
* 原假设：序列无自相关
corrgram log_return,lags(20)
* p值 < 0.05 → 存在自相关
wntestq log_return, lags(10)   // 滞后10期
wntestq log_return, lags(20)   // 滞后20期


* -------------- 4. ARCH效应检验（异方差检验）--------------
* 法2直接用官方命令检验（最简单）
reg log_return
estat archlm

import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\BTCUSD_H1.csv"

gen double datetime = clock(time, "YMDhms")
format datetime %tc
drop if datetime < td(12dec2017)
drop if missing(close, datetime)
sort datetime
* 只保留核心变量（日期+收盘价），简化数据
keep datetime close
order datetime close
tsset datetime  // 声明为时间序列（如果需要做时序分析）

* ==============================
*  2. 生成核心变量
* ==============================
* 2.1 小时级对数收益率（不依赖tsset，避免内存问题）
gen log_return = ln(close) - ln(close[_n-1])
replace log_return = . in 1  // 第一行无滞后数据，设为缺失
label var log_return "小时级对数收益率"

* 2.2 20小时滚动历史波动率（标准差）
gen vol_20hour = .
label var vol_20hour "20小时滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    summarize log_return in `start'/`i'
    replace vol_20hour = r(sd) in `i'
}

* 2.3 年化波动率（小时数据：√(252*24)=√6048）
gen vol_annual = vol_20hour * sqrt(6048)
label var vol_annual "20小时年化波动率"
drop if missing(log_return, vol_annual)

/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
twoway line close datetime, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return datetime, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual datetime, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Hour Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)

* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("BTCUSD Hourly Timing Analysis")

* 5）保存最终图片
graph export "three_combined.png", replace dpi(300)
import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\ETHUSD_H1.csv"

gen double datetime = clock(time, "YMDhms")
format datetime %tc
drop if date < td(12dec2017)
drop if missing(close, datetime)
sort datetime
* 只保留核心变量（日期+收盘价），简化数据
keep datetime close
order datetime close
tsset datetime  // 声明为时间序列（如果需要做时序分析）

* ==============================
*  2. 生成核心变量
* ==============================
* 2.1 小时级对数收益率（不依赖tsset，避免内存问题）
gen log_return = ln(close) - ln(close[_n-1])
replace log_return = . in 1  // 第一行无滞后数据，设为缺失
label var log_return "小时级对数收益率"

* 2.2 20小时滚动历史波动率（标准差）
gen vol_20hour = .
label var vol_20hour "20小时滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    summarize log_return in `start'/`i'
    replace vol_20hour = r(sd) in `i'
}

* 2.3 年化波动率（小时数据：√(252*24)=√6048）
gen vol_annual = vol_20hour * sqrt(6048)
label var vol_annual "20小时年化波动率"
drop if missing(log_return, vol_annual)
/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
* --------------------3张图放一起 --------------------
twoway line close datetime, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return datetime, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual datetime, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Hour Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)
	* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("ETHUSD Hourly Timing Analysis")
	import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\EURUSD_H1.csv"

gen double datetime = clock(time, "YMDhms")
format datetime %tc
gen date = dofc(datetime)
format date %td
* 现在可以正常比较了，删除 2017-12-12 之前的数据
drop if date < td(12dec2017)
drop if missing(close, datetime)
sort datetime
* 只保留核心变量（日期+收盘价），简化数据
keep datetime close
order datetime close
tsset datetime  // 声明为时间序列（如果需要做时序分析）

* ==============================
*  2. 生成核心变量
* ==============================
* 2.1 小时级对数收益率（不依赖tsset，避免内存问题）
gen log_return = ln(close) - ln(close[_n-1])
replace log_return = . in 1  // 第一行无滞后数据，设为缺失
label var log_return "小时级对数收益率"

* 2.2 20小时滚动历史波动率（标准差）
gen vol_20hour = .
label var vol_20hour "20小时滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    summarize log_return in `start'/`i'
    replace vol_20hour = r(sd) in `i'
}

* 2.3 年化波动率（小时数据：√(252*24)=√6048）
gen vol_annual = vol_20hour * sqrt(6048)
label var vol_annual "20小时年化波动率"
drop if missing(log_return, vol_annual)

/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
twoway line close datetime, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return datetime, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual datetime, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Hour Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)

* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("EURUSD Hourly Timing Analysis")

* 5）保存最终图片
graph export "three_combined.png", replace dpi(300)

import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\AUDUSD_H1.csv"

gen double datetime = clock(time, "YMDhms")
format datetime %tc
gen date = dofc(datetime)
format date %td
* 现在可以正常比较了，删除 2017-12-12 之前的数据
drop if date < td(12dec2017)
drop if missing(close, datetime)
sort datetime
* 只保留核心变量（日期+收盘价），简化数据
keep datetime close
order datetime close
tsset datetime  // 声明为时间序列（如果需要做时序分析）

* ==============================
*  2. 生成核心变量
* ==============================
* 2.1 小时级对数收益率（不依赖tsset，避免内存问题）
gen log_return = ln(close) - ln(close[_n-1])
replace log_return = . in 1  // 第一行无滞后数据，设为缺失
label var log_return "小时级对数收益率"

* 2.2 20小时滚动历史波动率（标准差）
gen vol_20hour = .
label var vol_20hour "20小时滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    summarize log_return in `start'/`i'
    replace vol_20hour = r(sd) in `i'
}

* 2.3 年化波动率（小时数据：√(252*24)=√6048）
gen vol_annual = vol_20hour * sqrt(6048)
label var vol_annual "20小时年化波动率"
drop if missing(log_return, vol_annual)
/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
* --------------------3张图放一起 --------------------
twoway line close datetime, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return datetime, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual datetime, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Hour Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)
	* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("AUDUSD Hourly Timing Analysis")

import delimited "C:\Users\86183\Desktop\ETH&BTC&其他国货币对美元汇率15分钟1小时1天\GBPUSD_H1.csv"

gen double datetime = clock(time, "YMDhms")
format datetime %tc
gen date = dofc(datetime)
format date %td
* 现在可以正常比较了，删除 2017-12-12 之前的数据
drop if date < td(12dec2017)
drop if missing(close, datetime)
sort datetime
* 只保留核心变量（日期+收盘价），简化数据
keep datetime close
order datetime close
tsset datetime  // 声明为时间序列（如果需要做时序分析）

* ==============================
*  2. 生成核心变量
* ==============================
* 2.1 小时级对数收益率（不依赖tsset，避免内存问题）
gen log_return = ln(close) - ln(close[_n-1])
replace log_return = . in 1  // 第一行无滞后数据，设为缺失
label var log_return "小时级对数收益率"

* 2.2 20小时滚动历史波动率（标准差）
gen vol_20hour = .
label var vol_20hour "20小时滚动标准差"

forvalues i = 20 / `=_N' {
    local start = `i' - 19
    summarize log_return in `start'/`i'
    replace vol_20hour = r(sd) in `i'
}

* 2.3 年化波动率（小时数据：√(252*24)=√6048）
gen vol_annual = vol_20hour * sqrt(6048)
label var vol_annual "20小时年化波动率"
drop if missing(log_return, vol_annual)
/* ===================== 3. 绘制三张时序图 ===================== */
* 设置绘图样式
set scheme s1color  // Stata 经典配色方案
* --------------------3张图放一起 --------------------
twoway line close datetime, ///
    lcolor(blue) lwidth(medthick) ///
    title("Closing Price") legend(off) name(g1, replace)

* 2）画第二张图 → 保存
twoway line log_return datetime, ///
    lcolor(red) lwidth(medthick) ///
    title("Logarithmic Return") legend(off) name(g2, replace)

* 3）画第三张图 → 保存
twoway line vol_annual datetime, ///
    lcolor(green) lwidth(medthick) ///
    title("20-Hour Rolling Historical Volatility") ///
    ylabel(, format(%9.2f)) ///
    legend(off) name(g3, replace)
	* 4）组合三张图（核心！）
graph combine g1 g2 g3, cols(1) title("GBPUSD Hourly Timing Analysis")