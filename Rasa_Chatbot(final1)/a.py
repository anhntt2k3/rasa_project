# import pandas as pd

# # Giai đoạn 1: Đọc chỉ 3 sheet cần ghép từ file Excel
# excel_file = "Data Trainning.xlsx"  # Đổi tên file tại đây
# sheets_to_merge = ["Câu hỏi và câu trả lời", "Tương tự", "Mẫu câu"]

# sheets = pd.read_excel(excel_file, sheet_name=sheets_to_merge)  # Đọc chỉ các sheet cần thiết

# # Giai đoạn 2: Hợp nhất các sheet thành một DataFrame
# df_list = []
# for sheet_name, df in sheets.items():
#     df["Sheet"] = sheet_name  # Thêm cột để biết dữ liệu từ sheet nào
#     df_list.append(df)

# merged_df = pd.concat(df_list, ignore_index=True)  # Ghép tất cả các sheet

# # Giai đoạn 3: Lưu file thành CSV
# csv_file = "merged_data.csv"
# merged_df.to_csv(csv_file, index=False)

# print(f"File CSV đã được tạo: {csv_file}")

# Hỏi nhiều, hỏi linh tình => Dùng LLM rewrite lại câu query theo dạng list, r dùng python parse ra rồi xử lý như bth