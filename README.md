![image](https://github.com/ldx-star/Bundle-Adjustment/assets/80197198/9c0930c7-1bb4-4b14-b97d-9e80b1e5b1d7)1、test_ba.txt文件包含了相机参数、三维点坐标、以及对应的观测点

2、使用高斯-牛顿法进行优化，优化公式如下（详细细节参考博文[牛顿法、高斯牛顿法](https://blog.csdn.net/holle_world_ldx/article/details/138225785)）：
$$X = X^{(k)}-\left(J(X)^TJ(X) + \mu_k I\right)^{-1}J(X)^Tr(X)$$

3、雅可比矩阵求解过程参考博文[捆绑调整](https://blog.csdn.net/holle_world_ldx/article/details/138442290?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22138442290%22%2C%22source%22%3A%22holle_world_ldx%22%7D)

4、结果
