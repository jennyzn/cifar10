package com.city945.cfar10;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;


import java.io.File;




public class MainActivity extends AppCompatActivity {

    public static String appRootDir;
    PictureGetter pt;
    ImageView inputImg;
    Button resText;

    public static final int REQUEST_CODE_CAMERA_WRITE = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initView();
        initAction();
    }
    private void initView(){
        inputImg = findViewById(R.id.input_image);
        resText = findViewById(R.id.result_text);
        pt = new PictureGetter(this, getWindowManager(), inputImg);

        //权限获取，一次性把相机跟存储的权限全获取了
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.CAMERA}, REQUEST_CODE_CAMERA_WRITE);
            // requestCode=1，随便设仅用于onRequestPermissionsResult的判断
        }else makeDir();//否则，已授权则建立根目录

    }

    private void initAction(){
        inputImg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pt.takePhoto();
            }
        });
        inputImg.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {
                pt.openGallery();
                return false;
            }
        });
        resText.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                resText.setText(Cifar10.className);
            }
        });
    }


    //文件读写权限
    private void makeDir(){
        try{
            appRootDir = getExternalFilesDir("./Picture").getAbsolutePath();//建立目录./Android/data/com.city945.cfar10/Picture
            File root = new File(appRootDir);
            if(!root.exists())root.mkdirs();
        }catch (SecurityException e){
            Toast.makeText(MainActivity.this,"error", Toast.LENGTH_LONG).show();
            e.printStackTrace();
        }
    }

    // 动态权限申请
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults){
        switch (requestCode){
            case REQUEST_CODE_CAMERA_WRITE:
                if(grantResults.length>0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)makeDir();
                else Toast.makeText(MainActivity.this,"You denied the permission", Toast.LENGTH_LONG).show();
                break;
            default:
        }
    }


    //相机
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        pt.myActyvityResult(requestCode,resultCode,data);// 返回数据的处理方法也在自身类中实现

        super.onActivityResult(requestCode, resultCode, data);
    }


}
