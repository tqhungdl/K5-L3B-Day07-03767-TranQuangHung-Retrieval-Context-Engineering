# Báo cáo Day 7 — Trần Quang Hưng

- MSSV: 2A202603767.
- Báo cáo dùng số liệu thực đã lưu trong `results/final/`; không dùng mock để báo chất lượng model.
- Trạng thái: đã đo retrieval và 85 lượt dev agent thật; test final và nhãn xem các bảng dưới. Review có AI hỗ trợ, cần người học xác nhận trước khi nộp.

## Dữ liệu và benchmark

- Nguồn: NIST, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, 01/2023; https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf. Theo manifest Day06: 48 trang vật lý, chọn 39 trang nội dung; 282 chunk. Trang là số trang vật lý PDF, bắt đầu từ 1.
- Quyền sử dụng theo manifest Day06: tài liệu liên bang NIST, public domain tại Hoa Kỳ, có ghi công; không suy rộng cho quyền bên thứ ba/mọi quốc gia.
- Dùng lại PostgreSQL/pgvector Day06, exact cosine search; strategy B: paragraph chunking, 500 ký tự, overlap 0. Embedding: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 chiều, normalized; query/document dùng cùng encoder, không prefix.
- Phát hiện JSON export cũ khác text/metadata trong database dù cùng ID. Đã xuất snapshot chuẩn sang `data/benchmark/chunks.json`; dense/hybrid kiểm tra toàn bộ ID, text, source/page và metadata trước khi chạy. Không so sánh lẫn snapshot cũ/mới.
- `data/benchmark/questions.json`: dev 15 có đáp án + 2 thiếu evidence; test 5 có đáp án + 2 thiếu evidence. Tất cả có answerable boolean; câu thiếu evidence có gold rỗng, verification_notes và kỳ vọng abstain. Các câu hỏi tổ chức giả định gần chủ đề risk/privacy/incidents/audit, nhưng corpus NIST không có hồ sơ nội bộ của tổ chức đó. Không suy ra thiếu evidence từ một lần top-k thất bại.
- Hit yêu cầu đúng source_id, trang, và evidence nguyên văn trong một chunk. Gold chỉ dùng sau retrieval để chấm; agent nhận question và history, không nhận gold/expected_answer/answerable.
- `validation-repairs.json` ghi hai sửa dữ liệu minh bạch: q18 rút suffix “Figure 2” bị cắt qua chunk, giữ nguyên câu hỏi/đáp án, sửa trước khi tính test; q08 bổ sung hành động safe cessation vào quote trong cùng chunk và bỏ điều kiện continuation không đủ evidence trong expected answer. q08 được kiểm tra giữ nguyên tập chunk hit trên toàn corpus, nên không thay đổi số retrieval đã đo. Hash gốc được giữ trong config/selection và các bản benchmark trong `data/benchmark/versions/`.

## Retrieval và thử nghiệm

Ba phương pháp: lexical TF-IDF; dense pgvector; hybrid RRF (rank constant 60, 20 candidates mỗi nhánh, rank bắt đầu từ 1). Không cộng trực tiếp TF-IDF score với cosine score. Không dùng filter cho bất kỳ phương pháp nào để giữ cùng điều kiện.

| Run | Split | Top-k | Budget ký tự | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median query + pack (ms) | Phí API retrieval |
|---|---|---:|---:|---|---|---|---:|---|
| retrieval-dev-dense | dev | 3 | 800 | 9/15 (60.00%) | 11/15 (73.33%) | 10/15 (66.67%) | 114.7985 | 0 USD |
| retrieval-dev-hybrid | dev | 3 | 800 | 10/15 (66.67%) | 13/15 (86.67%) | 11/15 (73.33%) | 117.2983 | 0 USD |
| retrieval-dev-hybrid-context1600 | dev | 3 | 1600 | 10/15 (66.67%) | 13/15 (86.67%) | 13/15 (86.67%) | 116.6038 | 0 USD |
| retrieval-dev-hybrid-top5 | dev | 5 | 1600 | 10/15 (66.67%) | 13/15 (86.67%) | 13/15 (86.67%) | 116.9866 | 0 USD |
| retrieval-dev-tfidf | dev | 3 | 800 | 10/15 (66.67%) | 15/15 (100.00%) | 14/15 (93.33%) | 4.7242 | 0 USD |
| retrieval-test-hybrid-context1600 | test | 3 | 1600 | 5/5 (100.00%) | 5/5 (100.00%) | 5/5 (100.00%) | 109.4469 | 0 USD |
| retrieval-test-tfidf | test | 3 | 800 | 5/5 (100.00%) | 5/5 (100.00%) | 5/5 (100.00%) | 4.7226 | 0 USD |

Budget tính cả header nguồn và ID; top-5 vẫn báo Hit@3 từ ba kết quả đầu. Hai câu thiếu evidence mỗi split được loại khỏi mẫu số retrieval. Lỗi được ghi riêng, không coi là chất lượng thấp của model. Model/index initialization tách riêng trong summary; số median trên không bao gồm thời gian load model. Phí API 0 USD của retrieval không bao gồm điện/phần cứng. Thời gian nhạy với cache/khởi động; không dùng latency của một lần chạy để kết luận hiệu năng tổng quát.

### Hai cấu hình cải thiện đã thử

1. Hybrid budget 800 → 1600, giữ top-3: giả thuyết giảm loại evidence khi pack context. Context Evidence Rate tăng 11/15 → 13/15; Hit@3 vẫn 13/15. Lớn hơn không tự động tốt hơn TF-IDF (14/15).
2. Hybrid budget 1600, top-3 → top-5: giả thuyết thêm evidence ngoài top-3. Context Evidence Rate vẫn 13/15; không có cải thiện trong benchmark này. Hit@3 không đổi vì cutoff so sánh vẫn 3. Đây là kết quả không cải thiện được báo đầy đủ.

### Phân tích lỗi dev

- q04: TF-IDF đưa heading “Part 2: Core and Profiles” trang 25 lên rank 1; đoạn GOVERN 1.1 trang 27 ở rank 2 và vào context. Hit@1 thất bại nhưng Hit@3/context đạt. Lý do khả dĩ: câu hỏi trùng từ với heading, không phải đoạn trả lời. Đây là lỗi ranking, chưa chứng minh đáp án agent sai.
- q07: gold trang 8 nằm rank 3 trong TF-IDF và bị budget 800 loại. Tuy nhiên, đọc context cho thấy rank 1 trang 25 đã nêu đủ GOVERN, MAP, MEASURE, MANAGE. Context Hit tự động = 0 là false negative tiềm năng của gold đơn đoạn; không được kết luận context thật sự thiếu bằng chứng. Giữ nguyên số liệu, ghi hạn chế và cần adjudication của người học.
- Dense thiếu gold trong top-3 ở q03/q07/q13/q15. Hybrid phục hồi một phần nhưng q07/q13 vẫn thiếu gold ngay cả với budget/top-k lớn hơn. Tăng budget không sửa được mọi lỗi ranking.

### Chọn cấu hình và test

Protocol được ghi trước lượt dev trong `results/final/protocol.json`: ưu tiên exact-gold Context Hit, rồi Hit@3, Hit@1, rồi median latency thấp hơn. Chọn **TF-IDF top-3, budget 800** trên dev. `selection.json` ghi thời điểm chốt và fingerprint; sửa gold để hợp lệ được ghi riêng, không đổi model/cấu hình hay chọn lại theo test.

Đây là **selection chỉ dựa trên retrieval**, chưa phải selection theo chất lượng câu trả lời agent. Baseline và cấu hình chọn trùng nhau nên chỉ có một lượt test TF-IDF, không coi là hai quan sát độc lập. Kết quả test: 5/5 ở cả ba retrieval metrics. Baseline test cũ đã được xem trước phiên này; không dùng các kết quả đó để chỉnh cấu hình mới. Cần công bố hạn chế held-out này, không tuyên bố một test hoàn toàn chưa từng thấy.

Đã đọc trực tiếp năm context test trong `retrieval-test-tfidf/context-reviews.json`: q18 (AI Model phase), q19 (procurement authority/acquisition), q20 (mất context và tác động), q21 (frequency/rationale), q22 (outcome-focused catalog). Cả năm đủ evidence và nguồn: mức 0 = 0, mức 1 = 0, mức 2 = 5. Đây là đánh giá AI-assisted của Codex dựa trên nội dung thực; người học cần xác nhận, không ghi thành nhãn human review đã hoàn tất.

## Agent và đánh giá

- Code dùng OpenAI Responses API structured output, model mặc định `gpt-5.4-mini`, đổi bằng OPENAI_MODEL. Key lấy từ môi trường/Day07 .env, không đưa vào artifact. Response đã lưu xác nhận snapshot gpt-5.4-mini-2026-03-17.
- Agent có thể chọn search hoặc answer; tối đa 4 searches, top-k/budget theo cấu hình. Ghi query, tham số, kết quả, context, câu trả lời, citation, lỗi và latency. Controller ghi response model và token usage khi API trả về; USD chưa biết để null, không tự nhận là 0.
- Context tích lũy qua các lượt, deduplicate theo ID và pack theo thứ tự; chỉ gửi context hiện tại, tránh nhân budget qua history. Hạn chế: chưa rerank xuyên các lượt; đoạn cũ có thể chặn evidence mới.
- Citation ID phải thuộc context và xuất hiện trong answer. Kiểm tra này không chứng minh semantic grounding; cần đọc từng citation. Thiếu citation/context hoặc vượt giới hạn search dẫn tới abstain; lỗi provider/store không fallback sang mock.

### Kết quả agent thật — nhãn AI-assisted

Đọc trực tiếp actual answer, final context và từng đoạn được cite; nhãn không suy ra chỉ từ ID hợp lệ. Các file `reviews.ai-assisted.json` ghi người review là Codex, phương pháp và hash trace. Đây là review có AI hỗ trợ, **người học cần kiểm tra/xác nhận**, không nhận là nhãn human review đã duyệt. Giữ nguyên `reviews.pending.json` để phân biệt dữ liệu chưa chấm với nhãn đã đề xuất.

| Run | Tool use | Số tool calls | Grounded answer 0/1/2; mean /2 | Citation hỗ trợ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| agent-dev-dense | 6/17 (35.29%) | 8 | 12/0/3; 0.400/2 | 3/4 (75.00%) | 4/4 (100.00%) | 2/2 (100.00%) | 11/15 (73.33%) |
| agent-dev-hybrid | 3/17 (17.65%) | 3 | 12/0/3; 0.400/2 | 3/3 (100.00%) | 3/3 (100.00%) | 2/2 (100.00%) | 12/15 (80.00%) |
| agent-dev-hybrid-context1600 | 7/17 (41.18%) | 7 | 10/0/5; 0.667/2 | 8/9 (88.89%) | 6/6 (100.00%) | 2/2 (100.00%) | 9/15 (60.00%) |
| agent-dev-hybrid-top5 | 4/17 (23.53%) | 4 | 12/0/3; 0.400/2 | 6/6 (100.00%) | 4/4 (100.00%) | 2/2 (100.00%) | 11/15 (73.33%) |
| agent-dev-tfidf | 5/17 (29.41%) | 5 | 11/0/4; 0.533/2 | 4/4 (100.00%) | 4/4 (100.00%) | 2/2 (100.00%) | 11/15 (73.33%) |
| agent-test-hybrid-context1600 | 1/7 (14.29%) | 1 | 5/0/0; 0.000/2 | 0/0 (không áp dụng) | 0/0 (không áp dụng) | 2/2 (100.00%) | 5/5 (100.00%) |
| agent-test-tfidf | 2/7 (28.57%) | 2 | 4/0/1; 0.400/2 | 1/1 (100.00%) | 1/1 (100.00%) | 2/2 (100.00%) | 4/5 (80.00%) |

Phân bố tool calls và mức context được giữ riêng, không thay bằng trung bình:

| Run | Số câu theo số search calls | Context 0/1/2 trên câu có đáp án | Context đủ evidence |
|---|---|---|---|
| agent-dev-dense | {'0': 11, '1': 5, '2': 0, '3': 1} | 12/0/3 | 3/15 (20.00%) |
| agent-dev-hybrid | {'0': 14, '1': 3} | 12/0/3 | 3/15 (20.00%) |
| agent-dev-hybrid-context1600 | {'0': 10, '1': 7} | 10/0/5 | 5/15 (33.33%) |
| agent-dev-hybrid-top5 | {'0': 13, '1': 4} | 12/0/3 | 3/15 (20.00%) |
| agent-dev-tfidf | {'0': 12, '1': 5} | 11/0/4 | 4/15 (26.67%) |
| agent-test-hybrid-context1600 | {'0': 6, '1': 1} | 5/0/0 | 0/5 (0.00%) |
| agent-test-tfidf | {'0': 5, '1': 2} | 4/0/1 | 1/5 (20.00%) |

Context agent ở đây là context thật sau các quyết định/query rewrite của LLM, chấm ngữ nghĩa. Nó khác context retrieval-only từ query gốc; không gộp hai loại số liệu. Ví dụ q07 có evidence thay thế ở trang 25 dù gold chuẩn nằm trang 8.

### Model, latency và chi phí

Model thực tế trong response: `gpt-5.4-mini-2026-03-17`. Mỗi file cost-estimate ghi model và usage gốc; các response đã review không có cached input. Ước tính theo giá standard 0.75 USD/triệu input tokens và 4.50 USD/triệu output tokens theo [OpenAI](https://openai.com/index/introducing-gpt-5-4-mini-and-nano/). Đây là ước tính token, **không phải hóa đơn**, chưa tính thuế hoặc điều chỉnh region/service tier; `service_cost_usd` trong output gốc vẫn null để không giả là billed cost.

| Run | API calls | Input tokens | Output tokens | Median end-to-end ms/câu | USD ước tính |
|---|---:|---:|---:|---:|---:|
| agent-dev-dense | 25 | 9510 | 1936 | 2020.24 | 0.015845 |
| agent-dev-hybrid | 20 | 6588 | 1496 | 2091.64 | 0.011673 |
| agent-dev-hybrid-context1600 | 24 | 10506 | 2176 | 2201.75 | 0.017671 |
| agent-dev-hybrid-top5 | 21 | 8405 | 1793 | 2366.20 | 0.014372 |
| agent-dev-tfidf | 22 | 7647 | 1633 | 2554.85 | 0.013084 |
| agent-test-hybrid-context1600 | 8 | 2741 | 483 | 2231.92 | 0.004229 |
| agent-test-tfidf | 9 | 3250 | 610 | 2975.67 | 0.005182 |

### Lỗi và đánh đổi quan sát trên dev

- Nhiều câu có đáp án bị từ chối **trước bất kỳ search nào**: history/context rỗng và event cuối `invalid_or_missing_citations`. Prompt “Always search before answering” chưa được enforce trong code. Cần phân biệt lỗi điều phối này với retriever không tìm được evidence.
- Dense q15: câu trả lời gán quy tắc ưu tiên theo sensitive data/human impact ở trang 13 cho MANAGE 1.2. Nguồn không hỗ trợ attribution; các yếu tố đúng cần có là impact, likelihood, available resources/methods. Chấm answer 0 và citation không hỗ trợ.
- Hybrid-context1600 q15: cite MANAGE 1 thành MANAGE 1.2 và không trả ba yếu tố; một citation khác hỗ trợ câu phụ về giảm tác động/human intervention nhưng không cứu được đáp án chính. Chấm answer 0.
- Hybrid-top5 q03: các citation hỗ trợ những câu phụ về PET/privacy tradeoffs, nhưng context không chứng minh kết luận chính về transparency. Vì vậy citation hợp lệ cho câu phụ không đồng nghĩa grounded answer đúng; chấm answer 0.
- Budget hybrid 800 → 1600 tăng số câu answer grade 2 từ 3/15 lên 5/15, giảm false abstention 12/15 → 9/15, nhưng tăng token/cost. Thêm top-5 không tăng chất lượng trong lượt này: grade 2 còn 3/15. Có thể do biến thiên query/decision của model; đây là một lần chạy, không phải kết luận nhân quả hay benchmark ổn định qua nhiều seed.

### Chọn cấu hình agent và trình tự test

Giữ nguyên `selection.json` cho kết luận retrieval-only (TF-IDF). Sau khi đọc toàn bộ dev agent, ghi `agent-selection-protocol.json` và `selection.agent.json`: chọn theo dev grounded-answer mean, context đủ evidence, tool use, citation support; tie-break bằng cost/latency. Chọn **hybrid, top-3, budget 1600, candidate-k 20, RRF constant 60**. Không đổi prompt hoặc tuning tham số từ test. Cấu hình này có 5/15 câu dev grade 2, so với 4/15 của agent TF-IDF.

Lượt TF-IDF agent test đầu tiên trong `results/api-runs/agent-test-tfidf` được giữ như **preliminary**, không trộn với lượt final. Baseline và cấu hình chọn được yêu cầu chạy lại cùng bộ test sau khi chốt agent selection, lưu trong `results/agent-final/`. Test đã từng được xem qua baseline cũ và preliminary run; công bố hạn chế này, không tuyên bố pristine held-out. Việc chọn cuối chỉ dùng dev và được ghi provenance.

### Ví dụ thực từ trace

- **Có evidence — q01**: No. The AI RMF can be used to prioritize risk, but it does not prescribe an organization's risk tolerance. Risk tolerance is described as the organization’s or AI actor’s readiness to bear risk to achieve its objectives, and it is highly contextual and use-case specific [ecfae9d34a360ab675aa8205d05cca27428dcd0239d225ea1f69f80dd64dc6d7]. Nguồn đối chiếu: `results/api-runs/agent-dev-tfidf/audit.md`.
- **Thiếu evidence — q23**: Không đủ bằng chứng để trả lời. Nguồn đối chiếu: `results/api-runs/agent-test-tfidf/audit.md`.
- **Lỗi attribution — q15**: MANAGE 1.2 says risk treatment should be prioritized based on the sensitivity of the data used to train the AI system and the potential impact of the system’s outputs on humans. It notes that higher priority may be appropriate when the system is trained on large datasets containing sensitive or protected data, such as personally identifiable information, or when its outputs have direct or indirect effects on people. [077531e7c7cf66ca4c6d88cfcfa898775293a6a9d8e7b44f1560e270a292ff65] Nguồn đối chiếu: `results/api-runs/agent-dev-dense/audit.md`.

Mọi labels/summary/usage và full evidence được lưu cùng từng run. Không dùng câu trả lời kỳ vọng thay actual output.

### Kết luận test final

- agent-test-hybrid-context1600: 0/5 câu có đáp án được trả lời đúng và đủ; false abstention 5/5 (100.00%); abstention đúng 2/2 (100.00%). Context mức 0/1/2: {'0': 5, '1': 0, '2': 0}.
- agent-test-tfidf: 1/5 câu có đáp án được trả lời đúng và đủ; false abstention 4/5 (80.00%); abstention đúng 2/2 (100.00%). Context mức 0/1/2: {'0': 4, '1': 0, '2': 1}.

Cấu hình chọn trên dev không cải thiện trên test. Không chọn lại hoặc thay prompt dựa trên kết quả này. Retrieval-only của cả baseline và hybrid đạt 5/5 context hit, nhưng agent thường dừng với invalid_or_missing_citations khi history chưa có search. Đây là lỗi điều phối: hướng phát triển tiếp theo là enforce một lượt retrieval ban đầu và đánh giá lại trên dev, sau đó dùng test mới nếu dùng kết quả test hiện tại để chỉnh. Không thay đổi hành vi đã benchmark trong phiên này.

## Tái lập và trạng thái nộp bài

- Lệnh dense: `python -X utf8 starter/benchmark.py --method dense --split dev --output results/manual-dev-dense`.
- Lệnh hybrid: `python -X utf8 starter/benchmark.py --method hybrid --split dev --output results/manual-dev-hybrid`.
- Toàn bộ suite: `starter/experiments.py dev`, `select-retrieval`, rồi `test`; các lệnh test yêu cầu frozen selection và fingerprint phù hợp. Xem RUNBOOK.md để dùng output root mới, tránh ghi đè evidence.
- Dataset có thể chia sẻ, config, protocol, per-query traces, summaries, context reviews và sửa gold nằm trong `data/benchmark/` và `results/final/`, ngoài thư mục starter bị ignore. Nguồn/giấy phép trong `data/benchmark/README.md`.
- Tests offline kiểm tra contract dense, filter lexical, RRF, snapshot conflict, split isolation, denominator loại câu thiếu evidence, missing key, parse/close và agent loop; không dùng test mock làm benchmark quality.
- AI hỗ trợ code, phân tích và báo cáo; người học chịu trách nhiệm kiểm tra nhãn, số liệu và nguồn theo RULES.md.

| Yêu cầu cơ bản | Trạng thái |
|---|---|
| Day06 thật, corpus/index tương thích | Đã kiểm tra snapshot và model identity |
| 3 phương pháp retrieval + trace/metrics | Đã chạy dev; test cho cấu hình đã chốt |
| 24 câu, đủ dev/test và thiếu evidence | Đã có, sửa evidence được ghi provenance |
| Ít nhất 2 thử nghiệm cải thiện + lỗi dev | Đã đo và phân tích, gồm thử nghiệm không cải thiện |
| Agent thật trên các cấu hình dev/test | Dev đã chạy đủ 5 cấu hình × 17 câu; test final kiểm tra trong results/agent-final |
| Grounded answer, citation support, abstention, ví dụ thực | Đã có review AI-assisted và ví dụ thực; cần người học xác nhận nhãn |
| Token/chi phí API | Đã có token usage và ước tính standard cost; billed cost cần đối chiếu invoice |
| Report, code, tests, run instructions | Đã có kết quả và hướng dẫn; kiểm tra lần cuối trước khi nộp |
| Push/nộp link và quyền người chấm | Chưa thực hiện; người học kiểm tra rồi nộp |

## Thử thách nâng cao (tùy chọn)

Chưa thực hiện. Corpus 48 trang không đạt yêu cầu tài liệu từ 100 trang; không báo điểm bonus hoặc benchmark bảng/ảnh/biểu đồ/multi-hop chưa chạy.
