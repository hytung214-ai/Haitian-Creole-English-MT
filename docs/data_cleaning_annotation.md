# 数据清洗标注说明

## 项目背景

本文档定义海地克里奥尔语（`hat`）到英语（`eng`）机器翻译项目中的数据清洗标注逻辑。我们的目标不是证明每一个被删除的句对都一定是错误的，而是建立一套清楚、可复现的训练数据条件，用来回答项目的核心问题：

**较小但更干净的训练集，是否能优于完整但噪声较多的训练集？**

清洗设计对应项目 proposal 和三周计划：

- `full`：保留原始训练数据，只去掉固定切出的开发集。
- `light_clean`：只删除明显、低争议的噪声。
- `risk_clean`：在 light cleaning 的基础上，进一步删除高风险样本。
- 同规模随机子集：后续生成与 `light_clean` 和 `risk_clean` 数据量相同的随机子集，用来区分“清洗效果”和“数据变少效果”。

## 总体标注原则

每一个平行句对都被看作一个训练样本。当句对符合某条规则时，我们给它分配一个或多个清洗标签。这些标签不是人工金标准标注，而是用于可复现过滤和后续分析的启发式标签。

本文档中的“词数”采用脚本中的简单 Unicode word token 计数：用正则 `\b\w+\b` 抽取 token。长度比例指：

```text
max(source_word_count, target_word_count) / min(source_word_count, target_word_count)
```

如果任一侧词数为 0，则长度比例视为无限大。

核心区别是：

- **Light cleaning 删除几乎肯定对平行 MT 训练有害或无效的数据。**
- **Risk-aware cleaning 删除不一定错误、但可能噪声较大、弱对齐、过于碎片化或过于特定领域的数据。**

短句需要谨慎处理：

```text
短本身不是噪声。
短 + 其他风险信号，才可能是噪声。
```

这一点很重要，因为短片段可能包含有效且有价值的翻译，例如人名、常用表达、命令、功能词搭配或领域术语。

## 数据条件 1：Full Data

`full` 是完整数据基线。它保留原始训练数据中除固定开发集之外的所有句对。

用途：

- 衡量模型使用最大训练集时的表现。
- 保留词汇覆盖、命名实体和领域多样性。
- 反映原始噪声数据分布。

做了什么：

- 从原始训练数据中固定切出 dev set。
- 官方 test set 原样复制，不做修改。
- 不做额外清洗。

## 数据条件 2：Light Cleaning

Light cleaning 是保守清洗。只有当噪声证据强、删除风险低时，才删除该句对。

### `empty_side`

定义：

- source 或 target 经过去除首尾空白后为空。
- 具体阈值：任一侧字符串长度为 `0`。

删除理由：

- 空源句或空目标句无法提供翻译关系。
- 可能导致分词或训练脚本出错。

清洗级别：

- Light cleaning。

过删风险：

- 很低。

### `identical_source_target`

定义：

- source 和 target 在归一化空白、HTML entity 和大小写后完全相同。
- 具体判定：先合并连续空白、解码 HTML entity、去除首尾空白、统一大小写；归一化后两侧字符串完全相同。

删除理由：

- 海地克里奥尔语和英语的完整句对通常不应该完全一样。
- 这类样本常见于复制的人名、UI 片段、未翻译文本或对齐错误。

清洗级别：

- Light cleaning。

过删风险：

- 低到中等。有些名字、缩写、数字或品牌短语确实可以合法地完全相同。但在本项目中，源目标完全相同是较强的噪声信号。

### `duplicate_pair`

定义：

- 同一个归一化后的 source-target 句对此前已经出现过。
- 具体判定：归一化方式与 `identical_source_target` 相同；如果 `(normalized_source, normalized_target)` 已经在前文出现，则后续副本删除。

删除理由：

- 重复句对会让常见网页片段获得过高权重，使训练分布失衡。
- 只保留一个副本可以更好地反映句对多样性。

清洗级别：

- Light cleaning。

过删风险：

- 低。删除的是重复副本，而不是唯一内容。

### `markup_or_html`

定义：

- source 或 target 中包含 HTML/XML 类标签、HTML entity 或 TMX 标记残留。
- 具体判定：任一侧匹配 HTML/XML 标签、HTML entity，或包含特定 TMX 残留字符串。

常见信号：

- `<tag>`、`<br>`、`</div>`
- `&nbsp;`、`&amp;`、`&#...;`
- `org.ldschurch.psd.mt.tmx`

删除理由：

- 这些标记不是自然语言翻译内容。
- 保留它们可能让模型学习输出技术性噪声。

清洗级别：

- Light cleaning。

过删风险：

- 低。即使某些标记包含结构信息，也不属于本项目的纯文本 MT 设置。

### `url_or_email`

定义：

- source 或 target 中包含 URL、网页地址或 email 地址。
- 具体判定：任一侧匹配 `http://`、`https://`、`www.` 或常见 email 格式。

删除理由：

- URL 和 email 通常跨语言直接复制，对学习翻译模式帮助有限。
- 它们常来自网页模板，而不是自然句子级翻译。

清洗级别：

- Light cleaning。

过删风险：

- 低。有些 URL 可能出现在有效句子中，但它们不是本实验重点。

### `encoding_replacement_char`

定义：

- source 或 target 中包含 Unicode 替换字符 `�`。
- 具体判定：任一侧出现字符 `�`。

删除理由：

- 这通常说明文本存在编码损坏或无法正确解码的问题。
- 损坏字符会影响 subword vocabulary 和模型输出。

清洗级别：

- Light cleaning。

过删风险：

- 很低。

### `extreme_length_ratio`

定义：

- source 和 target 的词数比例大于 `3.0`。
- 具体阈值：`length_ratio > 3.0`。

删除理由：

- 极端长度差异常说明弱对齐、内容缺失或源目标并非对应翻译。

清洗级别：

- Light cleaning。

过删风险：

- 中等。有些合法翻译天然长度不同，尤其是跨语言表达差异较大时。因此 light cleaning 中的阈值设得比较极端。

### `very_long_segment`

定义：

- source 或 target 任一侧超过 `100` 个词。
- 具体阈值：`max(source_word_count, target_word_count) > 100`。

删除理由：

- 很长的片段往往是段落、列表、网页拼接内容或切分错误。
- 它们可能主导训练 batch，并引入对齐噪声。

清洗级别：

- Light cleaning。

过删风险：

- 中等。有些长句对可能有效，但本项目主要关注句子级 MT。

### `punctuation_only`

定义：

- source 或 target 去除首尾空白后不包含字母或数字。
- 具体判定：任一侧非空，但没有任何 alphabetic character，也没有任何 digit。

删除理由：

- 纯标点不提供可学习的翻译内容。

清洗级别：

- Light cleaning。

过删风险：

- 很低。

### `number_only`

定义：

- source 或 target 包含数字，但不包含任何字母。
- 具体判定：任一侧非空，至少包含一个 digit，但不包含任何 alphabetic character。

删除理由：

- 纯数字行通常是 ID、价格、日期或表格片段，而不是可翻译句子。

清洗级别：

- Light cleaning。

过删风险：

- 低到中等。有些数字本身有意义，但纯数字句对对翻译建模帮助有限。

### `very_short_bad_fragment`

定义：

- 句对非常短，并且已经触发了至少一个 light-cleaning 层面的明显坏信号。
- 具体阈值：`min(source_word_count, target_word_count) <= 2`。
- 额外条件：必须已经触发至少一个其他 light-cleaning 标签，例如 `identical_source_target`、`markup_or_html`、`url_or_email`、`encoding_replacement_char`、`extreme_length_ratio`、`punctuation_only` 或 `number_only`。

删除理由：

- 1 到 2 个词的片段本身不一定是噪声，可能是有效短表达、名字、术语或常用词。
- 但如果非常短的片段同时带有明显坏信号，它通常更像网页碎片、表格残片、复制片段或无效训练样本。
- 因此 very short 不作为独立删除理由，只作为“非常短 + 明显坏”的附加标签。

清洗级别：

- Light cleaning。

过删风险：

- 低到中等。该规则不删除所有 very short 句对，因此比直接删除所有短句更保守。

## 数据条件 3：Risk-Aware Cleaning

Risk-aware cleaning 包含所有 light cleaning 规则，并额外加入更积极的规则，删除更可能有噪声的样本。这些样本不保证一定错误，因此在报告中应谨慎描述。

### `short_fragment_with_noise`

定义：

- 句对非常短，并且同时带有其他噪声或风险信号。
- 具体阈值：`min(source_word_count, target_word_count) <= 4`。
- 额外条件：该句对还必须已经触发至少一个其他清洗/风险标签；短本身不会单独导致删除。

删除理由：

- 短片段本身有歧义。它们可能是有用术语，也可能是菜单标签、标题、破碎对齐或复制片段。
- 只有当“短”与其他风险信号共同出现时才删除。

清洗级别：

- Risk-aware cleaning。

过删风险：

- 中等。该规则可能删除有用的短表达，因此不放入 light cleaning。

### `web_navigation_like`

定义：

- 句对包含较短的网页界面或导航类语言。
- 具体阈值：句对中出现网页导航关键词，并且 `max(source_word_count, target_word_count) <= 18`。

常见信号：

- `home`
- `login`
- `read more`
- `contact us`
- `privacy policy`
- `click here`
- `download`

删除理由：

- 这些片段常来自网站模板或导航栏，而不是有意义的句子翻译。
- 保留它们可能让模型学习网页模板文本。

清洗级别：

- Risk-aware cleaning。

过删风险：

- 中等。有些导航短语确实是合法翻译，但它们不是本项目主要关注的句子级 MT 内容。

### `product_or_technical_like`

定义：

- 句对包含产品目录、电商文本或技术规格信号。
- 具体判定：任一侧出现产品/技术关键词，或出现“数字 + 单位”的技术规格模式。

常见信号：

- `model`
- `specification`
- `price`
- `warranty`
- `voltage`
- `capacity`
- `shipping`
- `SKU`
- `mm`、`kg`、`kw`、`hz` 等单位

删除理由：

- 初步数据检查显示，语料中有很多目录式和技术参数式片段。
- 这类样本可能是弱翻译、模板化文本，或被数字和技术 token 主导。

清洗级别：

- Risk-aware cleaning。

过删风险：

- 中到高。技术术语可能有用，因此这个规则本质上是在测试：删除高风险领域文本是否有帮助，还是会损失有效信息。

### `title_like_fragment`

定义：

- 非常短的句对，看起来更像标题、标签或栏目名，而不是完整句子。
- 该规则应保持窄范围：长度短、无句末标点、两边都含有字母文本。
- 具体阈值：`max(source_word_count, target_word_count) <= 5`，并且 `min(source_word_count, target_word_count) <= 3`。
- 额外条件：source 和 target 都不以 `. ? ! : ;` 结尾，并且两侧都包含至少一个字母。

删除理由：

- 标题和标签可以是合法翻译，但通常缺少句法上下文，且容易来自网页片段。
- 删除它们可以测试更接近自然句子的训练数据是否提升 MT 质量。

清洗级别：

- Risk-aware cleaning。

过删风险：

- 中等。该规则必须保守，因为很多有效短翻译也没有句末标点。

### `suspicious_alignment`

定义：

- 句对具有中等偏高的长度比例，例如大于 `2.5`，或者一侧几乎包含另一侧，表现出复制或弱对齐迹象。
- 具体阈值 1：`length_ratio > 2.5`。
- 具体阈值 2：归一化后一侧字符串包含另一侧字符串，并且较短一侧词数 `<= 3`。

删除理由：

- 这些样本不一定像 light cleaning 中 `extreme_length_ratio` 那样明显坏，但仍有较高风险。
- 弱对齐会让模型学习不完整或不对应的翻译。

清洗级别：

- Risk-aware cleaning。

过删风险：

- 中等。有些合法翻译天然长度差异较大。

## 不应过度使用的规则

下面这些规则如果用得太激进，会损失有效数据：

- 不要删除所有短句。
- 不要因为句对 `<= 2` 个词就自动删除；只有当 very short 同时带有明显坏信号时，才进入 light cleaning。
- 不要删除所有标题式句对。
- 不要在 light cleaning 中删除所有产品或技术文本。
- 不要只因为英语流畅但海地克里奥尔语看起来奇怪，就认定样本错误。
- 除非有合格说话者或可靠来源确认，否则不要对海地克里奥尔语语法正确性作强判断。

## 报告中如何描述清洗

报告中应把这套清洗描述为启发式、实验性的清洗：

- Light cleaning 测试删除明显噪声是否有帮助。
- Risk-aware cleaning 测试删除高风险但可能有用的数据是否有帮助。
- 同规模随机子集测试提升是否来自更干净的数据，而不是单纯因为训练数据变少。

报告中应包含：

- 每个数据条件的句对数量。
- 每种清洗条件删除的比例。
- 各删除标签的计数。
- 每个主要标签下的一小组被删样例。
- 说明标签可能重叠，并且不是人工金标准标注。

## 当前数据切分与清洗统计

以下统计来自当前版本的 `scripts/prepare_mt_data.py`。脚本使用 `dev_ratio = 0.10`、`internal_test_ratio = 0.10`、`seed = 135`，先从原始训练集切分出 train/dev/internal test，再只对 train 部分进行清洗。官方 test set 不参与切分和清洗，只保留到最后评估使用。

原始数据规模：

| 数据来源 | 句对数 |
|---|---:|
| 原始 train | 665,033 |
| official test | 1,228 |

切分后数据规模：

| 数据 split | 句对数 | 说明 |
|---|---:|---|
| `full/train` | 532,027 | 约 80%，用于生成三种训练条件 |
| `dev` | 66,503 | 约 10%，用于开发阶段评估和调参 |
| `internal_test` | 66,503 | 约 10%，用于主要实验结果比较 |
| `official_test` | 1,228 | 官方 test set，最后额外评估 |

三种训练条件的清洗结果：

| 数据条件 | 输入句对数 | 保留句对数 | 删除句对数 | 删除比例 |
|---|---:|---:|---:|---:|
| `full` | 532,027 | 532,027 | 0 | 0.00% |
| `light_clean` | 532,027 | 466,288 | 65,739 | 12.36% |
| `risk_clean` | 532,027 | 278,879 | 253,148 | 47.58% |

同规模随机对照数据：

| 对照条件 | 抽样来源 | 匹配对象 | 句对数 | 随机种子 |
|---|---|---|---:|---:|
| `light_control` | `full/train` | `light_clean` | 466,288 | 1135 |
| `risk_control` | `full/train` | `risk_clean` | 278,879 | 1136 |

这两个对照集由 `scripts/create_matched_controls.py` 生成。它们不做清洗，只从 `full/train` 中随机抽取同样数量的句对，用来区分“清洗规则本身的效果”和“训练数据量减少的效果”。

`light_clean` 的删除原因计数：

| 标签 | 计数 |
|---|---:|
| `markup_or_html` | 26,134 |
| `very_short_bad_fragment` | 26,102 |
| `identical_source_target` | 25,696 |
| `duplicate_pair` | 9,707 |
| `very_long_segment` | 4,583 |
| `url_or_email` | 3,939 |
| `extreme_length_ratio` | 2,545 |
| `encoding_replacement_char` | 695 |
| `number_only` | 382 |
| `punctuation_only` | 262 |

`risk_clean` 的删除原因计数：

| 标签 | 计数 |
|---|---:|
| `title_like_fragment` | 162,454 |
| `product_or_technical_like` | 50,489 |
| `suspicious_alignment` | 35,048 |
| `short_fragment_with_noise` | 31,342 |
| `markup_or_html` | 26,134 |
| `identical_source_target` | 25,696 |
| `very_short_bad_fragment` | 24,374 |
| `web_navigation_like` | 14,196 |
| `duplicate_pair` | 4,878 |
| `very_long_segment` | 4,583 |
| `url_or_email` | 3,939 |
| `extreme_length_ratio` | 2,545 |
| `encoding_replacement_char` | 695 |
| `number_only` | 382 |
| `punctuation_only` | 262 |

注意：删除原因标签可以重叠。例如一个句对可能同时被标记为 `title_like_fragment` 和 `suspicious_alignment`。因此，各标签计数相加不等于总删除句对数。

## 建议输出目录

数据准备脚本建议生成：

```text
data/
  full/
    train.hat
    train.eng
  light_clean/
    train.hat
    train.eng
  risk_clean/
    train.hat
    train.eng
  light_control/
    train.hat
    train.eng
  risk_control/
    train.hat
    train.eng
  dev/
    dev.hat
    dev.eng
  internal_test/
    test.hat
    test.eng
  official_test/
    test.hat
    test.eng
```

报告目录建议包含：

```text
reports/data_preparation/
  condition_sizes.csv
  condition_stats.json
  matched_control_sizes.csv
  matched_control_sizes.json
  light_clean_removed.tsv
  light_clean_removed_samples.tsv
  risk_clean_removed.tsv
  risk_clean_removed_samples.tsv
  command_log.md
```

## 可直接用于最终报告的简短版本

我们使用两级启发式数据清洗。Light cleaning 删除低争议噪声，包括空句对、源目标完全相同的句对、重复句对、markup、URL、编码错误、极端长度比例、超长片段、纯标点片段和纯数字片段。对于非常短的句对，我们不因为长度本身删除；只有当较短一侧不超过 2 个词，并且同时触发其他明显坏信号时，才作为 `very_short_bad_fragment` 在 light cleaning 中删除。Risk-aware cleaning 在 light cleaning 的基础上进一步删除高风险样本，例如网页导航片段、产品或技术规格文本、标题式短片段以及中度可疑对齐。我们不会删除所有短句，因为短片段可能包含有效翻译；在 risk-aware cleaning 中，短本身也不是删除理由，只有当短片段同时带有其他风险信号时才删除。这些规则是启发式规则，不是人工金标准标注，因此我们会报告每类标签的删除数量，并抽样检查被删除的样本。
