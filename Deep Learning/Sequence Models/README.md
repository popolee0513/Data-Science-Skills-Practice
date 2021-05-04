Note
--------------------------
return_sequences=True 網絡輸出了每個timestep的h(t)

return_state=True 輸出是一個列表list，分別表示- 最後一個time step的hidden state - 最後一個time step的hidden state- 最後一個time step的cell state

RNN 梯度消失
--------------------------
RNN-based network is not always easy to learn.
The error surface is rough.

![https://ithelp.ithome.com.tw/upload/images/20191006/201204067w4Q4zFNH3.png](梯度消失與爆炸)

由舉例得知“w 在 0~0.99 之間變化，可能完全沒有造成影響；但 w 超過 1 後，只要一有影響，就是巨大的影響”
也就是會發生Gradient Vanish(梯度消失)或梯度爆炸的問題

#RNN中容易出現Gradient Vanishing是因為
1.在梯度在向後傳遞的時候，由於相同的矩陣相乘次數太多，梯度傾向於逐漸消失，導致後面的結點無法更新參數，整個學習過程無法正常進行

2.不適當的激活函數sigmoid，可改用relu或maxout
#而梯度爆炸的部分: 有一種暴力的方法就是，當梯度的大小超過某個閾值的時候，將其縮放到某個閾值
