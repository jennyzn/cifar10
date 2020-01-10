package com.city945.cfar10;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class Cifar10 {

    public static String[] IMAGENET_CLASSES = new String[]{
            "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    };
    public static String className = "请选择图片";
    private Context context;

    public Cifar10(Context context) {
        this.context = context;
    }
    public void pred(Bitmap bitmap){
        Module module = null;
        try{
            module = Module.load(assetFilePath(context, "ResNet.pt"));
        }catch (IOException e){
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            //finish();
            return;
        }

        // 预处理
        bitmap = Bitmap.createScaledBitmap(bitmap, 32,32,true);
        // 输入
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        // 运行
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        // 张量转java数组
        float[] scores = outputTensor.getDataAsFloatArray();
        // 从结果数组中找到最大得分
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        className = IMAGENET_CLASSES[maxScoreIdx];
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
