"""Build a reviewable paper bundle after the requested local runs finish."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path('outputs/herp_pilot_20260909')
mechanisms=Path('outputs/herp_mechanisms_20260909')
bundle=Path('outputs/herp_paper_bundle_20260909')
start=time.monotonic()
while not ((root/'suite_complete.json').exists() and (mechanisms/'p_32768/summary.json').exists()):
    if time.monotonic()-start>4*3600:raise TimeoutError('Suite not complete within four hours')
    time.sleep(10)
status=json.loads((root/'suite_complete.json').read_text())
if status['failures']:raise RuntimeError(status['failures'])
subprocess.run([sys.executable,'analysis/aggregate.py','--root',str(root),'--output-dir',str(bundle)],check=True)
subprocess.run([sys.executable,'analysis/plot_mechanisms.py','--root',str(mechanisms),'--output-dir',str(bundle)],check=True)
versions={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','matplotlib','gymnasium','mani-skill']}
(bundle/'environment.json').write_text(json.dumps(dict(python=sys.version,packages=versions,backend='physx_cpu',policy_device='cpu'),indent=2))
aggregation=json.loads((bundle/'aggregation.json').read_text())
assert aggregation['completed_runs']==36,aggregation['completed_runs']
table=(bundle/'RESULTS.md').read_text();mechanism_table=(bundle/'MECHANISMS.md').read_text()
report='''# HERP: kết quả chạy ngày 9/9/2026

## Phạm vi đã thực hiện

36 run ManiSkill: 2 task (PushCube, PickCube), 6 phương pháp, 3 seed, 32.768 tương tác huấn luyện/run. Mỗi checkpoint được đánh giá 50 episode từ ordinary reset, cùng seed đánh giá giữa các phương pháp. Tổng chi phí huấn luyện bằng 1.179.648 tương tác; probe và reference đã được tính vào số này. Đánh giá và mechanism có bộ đếm riêng.

Đây là pilot ngân sách nhỏ, không phải benchmark đã hội tụ. Bảng dưới đây chỉ chứa run hoàn tất. Thống kê qua seed không coi 50 episode đánh giá là 50 seed huấn luyện độc lập.

'''+table+'\n'+mechanism_table+'''
## Những gì có thể và chưa thể kết luận

- Tương quan sigma so sánh K=4 với K=64 futures độc lập từ cùng snapshot, trên 50 vùng ở mỗi checkpoint. Hai checkpoint thuộc cùng seed PushCube; không được diễn giải như hai seed độc lập.
- Tương quan p dùng 30 vùng, policy sao chép được cập nhật SGD một bước trên actor-head và log-standard-deviation, sau đó đo thay đổi return trên 50 episode cùng seed. Đây không phải kiểm định hiệu quả dài hạn của một full PPO update.
- Các khoảng bootstrap của mechanism là bất định qua vùng trong một checkpoint, không phải qua task hoặc training seed.
- Không thể khẳng định HERP vượt các baseline hoặc hội tụ từ riêng kết quả pilot. Nếu success thấp hoặc bằng 0, cần tăng ngân sách và kiểm tra PPO học task trước khi xây dựng claim chính cho paper.
- Wall time được đo khi có các job khác chạy đồng thời, gồm cả khởi tạo và đánh giá; không dùng để khẳng định compute overhead trong điều kiện cô lập.
- RND/Disagreement là phiên bản state-MLP trên cùng backbone PPO. Chưa tuning hệ số intrinsic reward, chưa tái lập benchmark chuẩn của code gốc.
- Chưa chạy bộ nhiều seed cho StackCube, PegInsertion, hay ablation Fisher/dot/occupancy riêng ở ngân sách dài. Các lựa chọn này đã được triển khai và kiểm tra đầu-cuối, nhưng không được tính thành kết quả nghiên cứu chưa chạy.

## File dùng cho paper

- `main_table.tex`: nội dung bảng LaTeX; `main_table.csv`: dữ liệu thống kê.
- `*_learning.pdf`: learning curves; vùng tô thể hiện ±1 SE qua training seed.
- `sigma_validation.pdf`: hai checkpoint sigma, `p_validation.pdf`: bốn estimator p.
- `correlations_bootstrap.json`: hệ số và khoảng bootstrap.
- Dữ liệu thô, config, checkpoint và SHA-256 mã nguồn nằm trong các thư mục run gốc.
- Quy trình và lệnh tái lập: `docs/HERP_REPRODUCIBILITY.md` tại gốc repository.
'''
(bundle/'REPORT_VI.md').write_text(report)
(bundle/'bundle_complete.json').write_text(json.dumps(dict(completed_runs=36,training_steps=36*32768,created_at=time.time()),indent=2))
print(str(bundle/'REPORT_VI.md'),flush=True)
