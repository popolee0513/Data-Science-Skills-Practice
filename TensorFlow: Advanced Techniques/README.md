Eager execution
---------------------------------------------------------------------------------------------------
Eager execution 又被稱為 Eager mode：可以使用者在建置模型時，就評估每個操作。而根據官方文件，Eager execution 有下列設計目標：

1.直覺性的介面：使用 Python 建立資料結構，執行環境可以依 Python 使用者常用函式庫建置，適合在小型資料和簡單的迴圈中使用。

2.更容易除錯：可使用 Python 標準函式庫的 debugger 除錯或檢查一個運算元的輸入和輸出

3.更自然的程式碼 flow control：直接用 Python 的 if...else 即可

4.Eager mode 的目標是在研發 prototype 設計階段時使用，若在生產階段，則需要在圖形模式下運行

AutoGraph

隨著eager execution的產生，衍生而成的需求就是AutoGraph了，也就是如何妥善的在eager與graph間轉換。而隨著session的消失、原生Python function的引入，帶來的就是tf.function了。
tf.function 的魔力在於將所有在eager下的計算，全部自動建好graph，也就是所謂的AutoGraph。因此可以引入圖的各項優勢，最直覺的就是運算速度上的提升。
