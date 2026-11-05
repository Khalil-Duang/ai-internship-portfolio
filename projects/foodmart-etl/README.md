# FoodMart 销售数据仓库 ETL

## 项目目标

将 FoodMart 销售主题中分散的客户、产品、促销、门店、时间和销售源表，整合为 6 个维度表和 1 个销售事实表，并通过可验证步骤检查 ETL 结果。

## 数据仓库结构

- `dim_customer`
- `dim_product`
- `dim_product_class`
- `dim_promotion`
- `dim_store`
- `dim_time`
- `fact_sales`

## 关键实现

1. 使用 `DropTable -> CreateTable -> 7 个数据流任务` 的顺序组织 SSIS 包。
2. 客户维度通过 `ISNULL(lname) + ' ' + fname` 处理空值后生成 `fullname`。
3. 三张销售事实表使用 `UNION ALL` 合并，统一列顺序并显式 `CAST` 字段类型。
4. 在事实数据流中派生 `gain = store_sales - store_cost`。
5. 保留建表、装载、抽样核对与结果验证 SQL。

## 验证方式

- 事实表行数应等于三张源销售表行数之和。
- `gain` 应逐行等于 `store_sales - store_cost`。
- 客户维度的 `fullname` 不应因姓或名缺失而整体变成 `NULL`。
- 源查询与目标字段的数据类型应一致。

## 运行

该项目依赖 SQL Server、SSIS 和 FoodMart 2000 源数据库。仓库只保留脱敏后的结构说明和建库脚本，不上传本机 `.dtsx` 连接串、机器名或真实数据库凭据。

