import TensorFlow

typealias TensorFloat = Tensor<Float>

struct NeuralNetwork {
    // 随机初始化权重, 原本范围在 0 <= n <= 1,  2n - 1 属于 [-1, 1]
    // 为什么要这样分, 因为对调试有帮助
    var weights: TensorFloat = 2 * TensorFloat(randomUniform: [3, 1]) - 1
    
    // 训练数据
    mutating func train(training_inputData: TensorFloat, training_outputData: TensorFloat, epoch: Int)
    {
        for _ in 0..<epoch {
            // 预测值
            let output = predict(inputs: training_inputData)
            // 误差值
            let error = training_outputData - output
            
            // 调整权重
            let adjusment = training_inputData.transposed() • (error * sigmoid_derivative(input: output))
            
            // 反向传播(Backpropagation): 将误差值再传回神经网络, 并调整权重.
            weights += adjusment
        }
    }
    
    // 预测数据
    func predict(inputs: TensorFloat) -> TensorFloat {
        // 权重就是通过这种方式来控制数据在神经网络间的流动
        // 此函数将返回预测结果
        // sigmoid 函数 是一个 S 型的曲线
        // 可以将数据归一化到 0 ~ 1 之间的概率, 用来帮助我们做出预测
        return sigmoid(inputs • self.weights)
    }
}

// sigmoid 的导数
// 描述了 sigmoid 曲线的梯度, 也就是变化率
func sigmoid_derivative (input: TensorFloat) -> TensorFloat {
    return input * (1 - input)
}

func main()
{
    // 初始化
    var network = NeuralNetwork()
    // 随机初始化权重
    print("初始化权重: ", network.weights)
    
    // 指定训练集
    // 三个输入 一个输出
    let training_inputData = TensorFloat([[0, 0, 1],
                                          [1, 1, 1],
                                          [1, 0, 1],
                                          [0, 1, 1]])
    let training_outputData = TensorFloat([[0, 1, 1, 0]]).transposed()

    // 开始训练
    network.train(training_inputData: training_inputData,
                  training_outputData: training_outputData, epoch: 10000)
    
    print("当前权重: ", network.weights)
    
    // 预测结果
    print("预测数据[[1, 0, 0]] -> ?", network.predict(inputs: TensorFloat([[1, 1, 1]])))
}

main()


