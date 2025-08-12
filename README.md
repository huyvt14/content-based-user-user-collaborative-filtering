## 1. Phương pháp Content-based filtering 
Đánh giá mức độ ưu tiên dựa trên 3 thuộc tính trên để gợi ý sản phẩm bao gồm: 
+ Danh mục (nam, nữ, trẻ em, em bé, sale) 
+ Dòng sản phẩm (quần, áo, giày, phụ kiện, sale, xu hướng)
+ Giá cả

## 2. Phương pháp Collaborative Filtering User-User 
Lựa chọn các đánh giá bằng sao của người dùng sau khi mua hàng làm đầu vào cho phương pháp Collaborative Filtering User-User. Mỗi lần mua hàng, mặc định đánh giá của người dùng cho sản phẩm là -1. Nếu người dùng đánh giá thì cập nhật giá trị đánh giá thực tế của người dùng.

## 3. Kết hợp Content-based filtering, Collaborative Filtering  bằng phương pháp Weighted Hybrid.
Tạo danh sách khuyến nghị từ riêng biệt từng phương pháp sau đó gộp lại.