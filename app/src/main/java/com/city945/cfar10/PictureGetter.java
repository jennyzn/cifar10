package com.city945.cfar10;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.view.WindowManager;
import android.widget.ImageView;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.SimpleDateFormat;
import java.util.Date;

import static android.app.Activity.RESULT_OK;

public class PictureGetter {

    private Context context;
    WindowManager wm;
    ImageView imageView;

    private Bitmap mBitmap;//从URI读取到的图片

    // 设置请求码
    private static final int REQUESTCODE_GALLERY = 10;//相册
    private static final int REQUESTCODE_TAKE_PHOTO = 20;//相机,对应MainActivity中的请求码
    private static final int REQUESTCODE_CUT = 30;//图片裁剪

    private Uri photoUri;

    public PictureGetter(Context context ,WindowManager wm, ImageView imageView) {
        this.context = context;
        this.wm = wm;
        this.imageView = imageView;
    }
    // 打开相机，对外接口
    public void takePhoto(){// 拍照需要先确定Uri，再裁剪
        Intent takeIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        // 这里拍照需要事先给出个Uri来知道存哪 https://blog.csdn.net/u011150924/article/details/71748464
        // 安卓7.0以上，应用间传Uri不能是File://Uri 而是Content://Uri Uri的获取不能用 Uri.fromFile
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        File mediaFile = new File(MainActivity.appRootDir + File.separator + "IMG_" + timeStamp + ".jpg");
        photoUri = FileProvider.getUriForFile(context.getApplicationContext(), "com.city945.cfar10.fileprovider", mediaFile);// Uri.fromFile(mediaFile);

        takeIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);// 所拍照片保存到 Uri处

        // 拍完照直接手动进入裁剪
        startPhotoZoom(photoUri);

        ((Activity)context).startActivityForResult(takeIntent, REQUESTCODE_TAKE_PHOTO);
    }

    // 打开相册,对外接口
    public void openGallery() {
        Intent picIntent = new Intent(Intent.ACTION_PICK,null);
        picIntent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,"image/*");
        ((Activity)context).startActivityForResult(picIntent,REQUESTCODE_GALLERY);//!!!为Activity的子类
    }

    public void myActyvityResult(int requestCode,int resultCode,Intent data){ //MainActivity中回调此方法，还是自己来处理数据
        if (resultCode == RESULT_OK) {
            switch (requestCode) {
                case REQUESTCODE_GALLERY:
                    //Toast.makeText(context.getApplicationContext(),"aaa", Toast.LENGTH_SHORT).show();
                    if (data == null || data.getData() == null)return;// 拍照后得到的data是null，在这里返回了
                    startPhotoZoom(data.getData());
                    break;
                case REQUESTCODE_CUT:
                    if (data!= null){
                        setPicToView();//会及时修改显示裁好的图
                    }
                    break;
            }
        }
    }

    // 启动裁剪窗口
    private void startPhotoZoom(Uri uri) {
        // 裁剪后的图片需要告诉Uri来自动保存
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        File mediaFile = new File(MainActivity.appRootDir + File.separator + "IMG_" + timeStamp + ".jpg");
        photoUri = Uri.fromFile(mediaFile);

        Intent intent = new Intent("com.android.camera.action.CROP");
        intent.setDataAndType(uri,"image/*");

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION); //添加这一句表示对目标应用临时授权该Uri所代表的文件
        }

        intent.putExtra("crop","true");

        // 切方形图片
        intent.putExtra("aspectX",1);
        intent.putExtra("aspectY",1);
        intent.putExtra("outputX",300);//像素宽度
        intent.putExtra("outputY",300);

        intent.putExtra("scale",true); //黑边
        intent.putExtra("scaleUpIfNeeded",true); //黑边

        intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);// 自动把裁好的照片放到函数开始时设置的Uri里去
        intent.putExtra("outputFormat", Bitmap.CompressFormat.JPEG.toString());

        intent.putExtra("return-data",false);

        ((Activity)context).startActivityForResult(intent, REQUESTCODE_CUT);
    }

    private void setPicToView() {
        try {
            mBitmap = BitmapFactory.decodeStream(context.getContentResolver().openInputStream(photoUri));
            imageView.setImageBitmap(mBitmap);
            Cifar10 cifar10 = new Cifar10(context);
            cifar10.pred(mBitmap);
            // TODO 把mBitmap返回做处理去

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
