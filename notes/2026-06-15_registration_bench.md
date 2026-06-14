# 登録手法 比較ベンチマーク（2026-06-15）

## 結論

合成ボックスコーナー（3直交面、7500点）を小さくミスアラインメント（yaw 0.03 rad ＋ 並進 ~0.02 m）させて4手法で復元:

| 手法 | 復元精度（probe 誤差 m） | 速度（中央値） | 所見 |
|------|------------------------|--------------|------|
| ICP (point-to-point) | 0.0196 | ~147 ms | 平面上は収束が遅く精度も最下位 |
| **point-to-plane ICP** | 0.00073 | **~6.5 ms** | **速度/精度のバランス最良** |
| GICP | **0.00057** | ~26 ms | **最高精度**だが per-point 共分散推定で最も重い |
| NDT | 0.00079 | ~8.7 ms | 高速かつ高精度（ボクセル分布＋LM） |

- **point-to-plane**: ターゲット法線を使い1反復で大きく収束 → 最速クラス＆高精度。平面が多いシーンの既定候補。
- **GICP**: source/target 双方の分布を考慮し最高精度。共分散を全点で推定するぶん遅い。
- **NDT**: ターゲットをボクセル分布化、Levenberg-Marquardt で安定。GICP に近い精度を 1/3 の時間で。
- **point-to-point ICP**: 平面主体だと点対点対応の収束が遅く、反復上限まで回りやすい。

## 計測条件

| 項目 | 内容 |
|------|------|
| ベンチコマンド | `cargo bench -p spatialrust-registration --features register-icp,register-icp-point-to-plane,register-gicp,register-ndt --bench registration` |
| 計測設定 | criterion `--warm-up-time 0.5 --measurement-time 1.5 --sample-size 10`（簡易計測） |
| 入力 | ボックスコーナー 50×50×3面 = 7500 点 |
| ミスアラインメント | yaw 0.03 rad ＋ 並進 (0.02, -0.015, 0.01) |
| 共通設定 | max_correspondence_distance=0.3, max_iterations=40（NDT は resolution=0.2, iters=50）|
| 精度指標 | 復元変換 ∘ ミスアラインメントを probe 点 (0.4,0.5,0.3) に適用した残差ノルム |

## 注意

- 数値は単一の合成シーン・小ミスアラインメントでの相対比較。実データ・大ミスアラインメントでは傾向が変わり得る（特に NDT/GICP は初期値依存）。
- point-to-point ICP の絶対時間は反復上限（40）に張り付くため大きめ。早期収束しきい値の調整で短縮可能。
