# 時系列データの定常性確認
時系列データの定常性を確認することでデータの性質を調べることができます。

# 定常性を確認する意味
時系列データの種類は大きく分けて以下の３つに分類できると考えられます。
+ 定常過程
+ 単位根過程
+ その他の生成過程

時系列データをデータ解析する際には、時系列データの種類を明確にして解析を進める必要があります。

## サンプルデータに関して
今回使用したサンプルデータのAirPassengers.csvに対して何も処理を施さずにADF検定を実施すると定常過程でも単位根過程でもありませんでした。
これに対して適切な判断を下すためには季節調整を施すなどの前処理が必要です。
ちなみに定常過程の代表的な時系列データとしてホワイトノイズがあります。
