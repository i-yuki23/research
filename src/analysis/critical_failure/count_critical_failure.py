"""
致命的に間違っているタンパク質がどのくらいあるのかを調べる

- 致命的に間違っているタンパク質とは、
    - リガンド原子と最近接のnon-replace水分子(予測)の距離がある閾値以内のもの

手順
1. 予測した水分子のpdbファイルを作成(replace、non-replaceで分ける)
2. 各リガンドの原子と最近説のnon-replace水分子の距離を計算(原子タイプを考慮して、距離が閾値以内のものの数をカウントする)
3. 0個のものがちゃんと予測できているたんぱく質
"""