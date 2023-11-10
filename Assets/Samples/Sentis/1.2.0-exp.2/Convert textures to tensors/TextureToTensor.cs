using UnityEngine;
using Unity.Sentis;
using UnityEngine.Assertions;

public class TextureToTensor : MonoBehaviour
{
    // 8x8 red texture
    [SerializeField]
    Texture2D texture;
    TensorFloat inputTensor;
    ITensorAllocator allocator;
    Ops ops;

    void Start()
    {
        // // tensor dimensions are taken from texture
        // using TensorFloat tensor = TextureConverter.ToTensor(texture);
        // Assert.AreEqual(tensor.shape, new TensorShape(1, 4, 8, 8));
        //
        // // specifying channel number truncates the channels from the texture
        // using TensorFloat tensorRGB = TextureConverter.ToTensor(texture, channels: 3);
        // Assert.AreEqual(tensorRGB.shape, new TensorShape(1, 3, 8, 8));
        //
        // // specifying width and/or height resamples the texture linearly
        // using TensorFloat tensor16X24 = TextureConverter.ToTensor(texture, width: 4, height: 12);
        // Assert.AreEqual(tensor16X24.shape, new TensorShape(1, 4, 12, 4));
        //
        // // for more complex conversions use a TextureTransform, the defaults are as above
        // using TensorFloat tensorT = TextureConverter.ToTensor(texture, new TextureTransform());
        // Assert.AreEqual(tensorT.shape, tensor.shape);
        //
        // // RGB tensor from transform
        // using TensorFloat tensorTRGB = TextureConverter.ToTensor(texture, new TextureTransform().SetDimensions(channels: 3));
        // Assert.AreEqual(tensorTRGB.shape, tensorRGB.shape);
        //
        // // linear sampled tensor from texture
        // using TensorFloat tensorT16X24 = TextureConverter.ToTensor(texture, new TextureTransform().SetDimensions(width: 4, height: 12));
        // Assert.AreEqual(tensorT16X24.shape, tensor16X24.shape);
        //
        // // alternative tensor layout
        // using TensorFloat tensorTNHWC = TextureConverter.ToTensor(texture, new TextureTransform().SetTensorLayout(TensorLayout.NHWC));
        // Assert.AreEqual(tensorTNHWC.shape, new TensorShape(1, 8, 8, 4));
        //
        // // explicit alternative tensor layout
        // using TensorFloat tensorTNHCW = TextureConverter.ToTensor(texture, new TextureTransform().SetTensorLayout(0, 2, 1, 3));
        // Assert.AreEqual(tensorTNHCW.shape, new TensorShape(1, 8, 4, 8));
        //
        // // set tensor 0, 0 from bottom left of texture rather than default top left
        // using TensorFloat tensorTBottomLeft = TextureConverter.ToTensor(texture, new TexureTransform().SetCoordOrigin(CoordOrigin.BottomLeft));
        //
        // // swizzle color channels of texture using preset
        // using TensorFloat tensorTBGRA = TextureConverter.ToTensor(texture, new TextureTransform().SetChannelSwizzle(ChannelSwizzle.BGRA));
        // // make the tensors readable (move to CPU) before accessing with indices
        // tensor.MakeReadable();
        // tensorTBGRA.MakeReadable();
        // Assert.AreEqual(tensorTBGRA[0, 2, 0, 0], tensor[0, 0, 0, 0]);
        //
        // // swizzle color channels of texture explicitly to all sample from Red channel in texture
        // using TensorFloat tensorTRRRR = TextureConverter.ToTensor(texture, new TextureTransform().SetChannelSwizzle(0, 0, 0, 0));
        // // make the tensors readable (move to CPU) before accessing with indices
        // tensor.MakeReadable();
        // tensorTRRRR.MakeReadable();
        // Assert.AreEqual(tensorTRRRR[0, 3, 0, 0], tensor[0, 0, 0, 0]);
        //
        // // chain transform settings together
        // using TensorFloat tensorTChained = TextureConverter.ToTensor(texture, new TextureTransform().SetDimensions(channels: 3).SetCoordOrigin(CoordOrigin.BottomLeft).SetChannelSwizzle(ChannelSwizzle.BGRA));
    }
    
    void Update()
    {
        // Create a one-dimensional input tensor with four values.
        inputTensor = new TensorFloat(new TensorShape(4), new[] { -2.0f, 1.0f, -0.2f, 0.0f });

        // Create an allocator.
        allocator = new TensorCachingAllocator();

        // Create an Ops object. The object uses Sentis compute shaders to do operations on the GPU.
        ops = WorkerFactory.CreateOps(BackendType.GPUCompute, allocator);

        // Run the ArgMax operator on the input tensor.
        using TensorFloat outputTensor = ops.ThresholdedRelu(inputTensor, 0.0f);
        outputTensor.MakeReadable();

        Debug.Log("outputTensor[0]: " + outputTensor[0] + " outputTensor[1]: " + outputTensor[1] + " outputTensor[2]: " + outputTensor[2] );
    }
    
}
