-- FoodMart 销售数据仓库建库脚本
-- 将 <YOUR_SQL_SERVER_INSTANCE> 和 <RETAIL_DW_NAME> 替换为本地环境值。

USE master;
GO

IF DB_ID('<RETAIL_DW_NAME>') IS NULL
BEGIN
    CREATE DATABASE [<RETAIL_DW_NAME>];
END
GO

-- 说明：
-- 1. FoodMart 源库需自行附加，数据库名使用 Foodmart2000。
-- 2. 7 张仓库表由 SSIS 包的 DropTable / CreateTable 任务创建。
-- 3. ETL 完成后应核对事实表行数、gain 派生列和客户姓名拼接结果。
