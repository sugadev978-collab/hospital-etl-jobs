
#Hospital-etl-job

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext 
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, lit, current_timestamp, to_date, year, month
from pyspark.sql.types import DoubleType, IntegerType

# =========================
# PARAMETERS
# =========================

args = getResolvedOptions(sys.argv, ["JOB_NAME", "SOURCE_BUCKET"])
source_bucket = args["SOURCE_BUCKET"]

# =========================
# SPARK SESSION
# =========================

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)
spark.conf.set("spark.sql.shuffle.partitions", "8")
print("===== HOSPITAL ETL JOB STARTED =====")

# =========================
# S3 PATHS
# =========================

patients_path = f"s3://{source_bucket}/patients/"
doctors_path  = f"s3://{source_bucket}/doctors/"
billing_path  = f"s3://{source_bucket}/billing/"
trusted_path  = f"s3://{source_bucket}/trusted/hospital_billing/"
rejected_path = f"s3://{source_bucket}/rejected/hospital_billing/"

# =========================
# READ DATA
# =========================

print("Reading data from S3...")

patients_df = spark.read.option("header", True).option("inferSchema", True).csv(patients_path)
doctors_df  = spark.read.option("header", True).option("inferSchema", True).csv(doctors_path)
billing_df  = spark.read.option("header", True).option("inferSchema", True).csv(billing_path)

print("Data Loaded")
print("Patients:", patients_df.count())
print("Doctors:",  doctors_df.count())
print("Billing:",  billing_df.count())

# =========================
# REMOVE DUPLICATES
# =========================

patients_df = patients_df.dropDuplicates(["patient_id"])
doctors_df  = doctors_df.dropDuplicates(["doctor_id"])
billing_df  = billing_df.dropDuplicates(["bill_id"])

# =========================
# TYPE CASTING
# =========================

billing_df = billing_df.withColumn("consultation_fee", col("consultation_fee").cast(DoubleType()))
billing_df = billing_df.withColumn("medicine_cost",    col("medicine_cost").cast(DoubleType()))
billing_df = billing_df.withColumn("room_charges",     col("room_charges").cast(DoubleType()))
billing_df = billing_df.withColumn("bill_date",        to_date(col("bill_date")))

# =========================
# BASIC VALIDATION
# =========================

valid_df = billing_df.filter(
    (col("bill_id").isNotNull())     &
    (col("patient_id").isNotNull())  &
    (col("doctor_id").isNotNull())   &
    (col("consultation_fee") > 0)    &
    (col("medicine_cost") >= 0)      &
    (col("room_charges") >= 0)       &
    (col("bill_date").isNotNull())
)

rejected_basic_df = billing_df.filter(
    (col("bill_id").isNull())          |
    (col("patient_id").isNull())       |
    (col("doctor_id").isNull())        |
    (col("consultation_fee") <= 0)     |
    (col("bill_date").isNull())
).withColumn("rejection_reason", lit("Basic validation failed"))

# =========================
# JOIN DIMENSIONS
# =========================

joined_df = valid_df \
    .join(patients_df, "patient_id", "left") \
    .join(doctors_df,  "doctor_id",  "left")

# =========================
# FINAL VALIDATION
# =========================

final_valid_df = joined_df.filter(
    (col("patient_name").isNotNull()) &
    (col("doctor_name").isNotNull())
)

rejected_join_df = joined_df.filter(
    (col("patient_name").isNull()) |
    (col("doctor_name").isNull())
).withColumn("rejection_reason", lit("Dimension lookup failed"))

# =========================
# FINAL TRANSFORM
# =========================

final_df = final_valid_df \
    .withColumn("total_bill",
        col("consultation_fee") + col("medicine_cost") + col("room_charges")) \
    .withColumn("bill_year",         year(col("bill_date"))) \
    .withColumn("bill_month",        month(col("bill_date"))) \
    .withColumn("etl_inserted_time", current_timestamp()) \
    .select(
        "bill_id",
        "patient_id",
        "patient_name",
        "city",
        "doctor_id",
        "doctor_name",
        "specialization",
        "department",
        "consultation_fee",
        "medicine_cost",
        "room_charges",
        "total_bill",
        "bill_date",
        "bill_year",
        "bill_month",
        "etl_inserted_time"
    )

# =========================
# COUNTS
# =========================

print("Valid records:",    final_df.count())
print("Rejected records:", rejected_basic_df.count() + rejected_join_df.count())

# =========================
# WRITE OUTPUT
# =========================

print("Writing Trusted Data...")

final_df.write \
    .mode("overwrite") \
    .partitionBy("bill_year", "bill_month") \
    .parquet(trusted_path)

print("Writing Rejected Data...")

rejected_basic_df.select(
    "bill_id",
    "patient_id",
    "doctor_id",
    "consultation_fee",
    "medicine_cost",
    "room_charges",
    "bill_date",
    "rejection_reason"
).unionByName(
    rejected_join_df.select(
        "bill_id",
        "patient_id",
        "doctor_id",
        "consultation_fee",
        "medicine_cost",
        "room_charges",
        "bill_date",
        "rejection_reason"
    )
).write \
    .mode("overwrite") \
    .parquet(rejected_path)

print("===== HOSPITAL ETL JOB COMPLETED =====")

job.commit()