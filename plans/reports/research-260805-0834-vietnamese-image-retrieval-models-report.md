---
type: research
status: complete
conducted_at: 2026-08-05
scope: AIC 2026 Request 1 Vietnamese text-to-image retrieval
---

# Nghiên cứu model retrieval tiếng Việt cho Request 1

## Tóm tắt

OpenAI CLIP ViT-B/32 hiện tại **có thể nhận chuỗi Unicode tiếng Việt nhưng không hỗ trợ tiếng Việt đủ tin cậy**. Thử nghiệm project đã cho kết quả English tốt, Vietnamese trực tiếp kém. Đây là giới hạn text encoder/training data, không phải lỗi index.

Lựa chọn đầu tiên: benchmark `sentence-transformers/clip-ViT-B-32-multilingual-v1`. Model này ánh xạ văn bản đa ngôn ngữ vào không gian ảnh CLIP ViT-B/32, output 512 chiều; có khả năng reuse index hiện tại. Lựa chọn đối chứng: dịch Việt → Anh rồi dùng exact OpenAI CLIP encoder hiện tại. Chỉ rebuild index với PhoCLIP/SigLIP 2 nếu hai hướng rẻ hơn không đạt metric.

## Tiêu chí

- Retrieval tiếng Việt trên video/keyframe AIC.
- Tương thích index OpenAI CLIP ViT-B/32 hiện có.
- Offline Kaggle, GPU T4/P100.
- License dùng được cho cuộc thi.
- Benchmark theo R@1/5/20/50/100; không chọn model theo demo cảm tính.

## Kết quả

| Phương án | Tiếng Việt | Reuse index 512-D hiện tại | Chi phí | Khuyến nghị |
|---|---:|---:|---:|---|
| Raw OpenAI CLIP ViT-B/32 | Yếu/không bảo đảm | Có | Thấp | Chỉ làm control |
| `clip-ViT-B-32-multilingual-v1` | Có, multilingual | Có khả năng cao | Thấp | Thử đầu tiên |
| M-CLIP text encoder tương ứng ViT-B/32 | Có, tùy checkpoint/language list | Có khả năng | Thấp-vừa | Phương án dự phòng |
| Việt → Anh + current CLIP | Phụ thuộc MT | Có | Vừa; thêm latency | Đối chứng bắt buộc |
| PhoCLIP/Vietnamese-specific CLIP | Thiết kế cho tiếng Việt | Không | Rebuild index | Chỉ benchmark sau |
| SigLIP 2 multilingual | Multilingual | Không | Rebuild index; nặng hơn | Candidate chất lượng |
| Jina CLIP v2 | Multilingual | Không | Rebuild index | License non-commercial cần kiểm tra |

### 1. OpenAI CLIP hiện tại

- Tokenizer vẫn mã hóa được tiếng Việt; điều này không đồng nghĩa model hiểu tiếng Việt.
- Training/model card không cam kết Vietnamese retrieval.
- Evidence project: cùng ý nghĩa, English query trả kết quả phù hợp; Vietnamese query trực tiếp kém.
- Không nên bỏ dấu hoặc thay tokenizer riêng: làm lệch text tower đã train.

### 2. Candidate không rebuild index

`sentence-transformers/clip-ViT-B-32-multilingual-v1` được distill để text đa ngôn ngữ khớp không gian `clip-ViT-B-32`. Ưu điểm:

- output 512-D;
- model card công bố 50+ ngôn ngữ;
- dùng cùng CLIP image space;
- chỉ thay query text encoder;
- phù hợp Kaggle.

Cần gate thực nghiệm: dimension 512 chưa chứng minh compatibility. Chạy cùng Vietnamese labeled queries trên exact index; kiểm tra top-k và R@k.

M-CLIP dùng ý tưởng tương tự: multilingual text encoder được align với CLIP visual space. Chỉ dùng checkpoint ghi rõ visual backbone OpenAI ViT-B/32 và Vietnamese nằm trong language coverage.

### 3. Dịch Việt → Anh

Pipeline:

```text
Vietnamese query → offline MT vi-en → current pinned OpenAI CLIP → current index
```

Ưu điểm: tận dụng English retrieval đã xác minh; không rebuild index. Nhược điểm: mất chi tiết, negation, tên riêng; latency/model thứ hai. Nên lưu aggregate latency/provenance, không serialize raw query. Model MT/license phải pin theo exact revision trước dùng.

### 4. Candidate cần rebuild index

- **PhoCLIP/Vietnamese-specific CLIP:** phù hợp ngôn ngữ nhất về thiết kế. Cần xác minh official checkpoint, license, benchmark; phải encode lại 177.321 ảnh vì vector space khác.
- **SigLIP 2:** multilingual vision-language model hiện đại; mạnh hơn về broad retrieval/understanding. Không tương thích OpenAI CLIP index dù dimension có thể trùng.
- **Jina CLIP v2:** multilingual, context dài; phải rebuild. Model card dùng license non-commercial; chưa nên đưa vào pipeline cuộc thi trước khi xác nhận quyền sử dụng.

## Cải tiến OpenAI CLIP

Thứ tự YAGNI:

1. Multilingual text encoder aligned với frozen OpenAI CLIP image space.
2. Offline Vietnamese-English translation.
3. Score fusion hai nhánh nếu từng nhánh có gain bổ sung.
4. Chỉ khi off-the-shelf thất bại: distill Vietnamese text encoder theo English CLIP embeddings bằng parallel VI-EN text.
5. Cuối cùng: fine-tune/rebuild image-text model bằng paired Vietnamese data hợp lệ.

Không làm ngay: tokenizer hack, bỏ dấu, synthetic captions, LVLM, joint fine-tuning, nhiều vector DB.

## Benchmark tối thiểu

Tạo private set 50–100 query tiếng Việt, phủ:

- người/hành động;
- vật thể/thuộc tính/màu;
- địa điểm/cảnh;
- thời gian/thời tiết;
- chữ viết/tên riêng;
- negation và quan hệ nhiều đối tượng.

Mỗi query có relevant `(video_id, frame_id)`; thêm English translation do người kiểm tra. So sánh cùng index/config:

1. raw Vietnamese + OpenAI CLIP;
2. multilingual CLIP text encoder;
3. human English translation + OpenAI CLIP — upper-bound translation;
4. machine translation + OpenAI CLIP.

Report: R@1/5/20/50/100, Final Score, p50/p95 encoding latency. Promotion gate đề xuất: Final Score tăng rõ ràng trên held-out set; không regression lớn trên English controls.

## Khuyến nghị triển khai

Increment kế tiếp nhỏ nhất:

- thêm query backend `multilingual-clip` riêng;
- pin exact model revision;
- giữ current OpenAI CLIP backend;
- không rebuild index;
- chạy benchmark 4 nhánh trên private paired set;
- chỉ sau kết quả mới quyết định translation hoặc rebuild bằng model khác.

## Nguồn

- [Sentence Transformers multilingual CLIP model](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)
- [Sentence Transformers image search documentation](https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html)
- [Multilingual-CLIP repository](https://github.com/FreddeFrallan/Multilingual-CLIP)
- [Google SigLIP 2 model card](https://huggingface.co/google/siglip2-base-patch16-224)
- [SigLIP 2 paper](https://arxiv.org/abs/2502.14786)
- [Jina CLIP v2 model card](https://huggingface.co/jinaai/jina-clip-v2)
- [OpenAI CLIP repository](https://github.com/openai/CLIP)

## Câu hỏi chưa giải quyết

- Chưa có public benchmark trực tiếp trên query tiếng Việt và dataset AIC hiện tại.
- Official PhoCLIP checkpoint/license cần xác minh trước khi đưa vào shortlist thực thi.
- Quyền dùng artifact AIC25 cho AIC2026 vẫn cần BTC xác nhận.
