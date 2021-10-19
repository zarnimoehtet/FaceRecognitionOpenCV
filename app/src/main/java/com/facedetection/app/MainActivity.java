package com.facedetection.app;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.CardView;
import android.view.View;
import android.widget.Button;
import android.widget.Switch;
import android.widget.TextView;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {


    CardView function;
    TextView action;
    Switch onOff;
    boolean isOn = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        function = findViewById(R.id.actionCard);
        action = findViewById(R.id.activateText);
        onOff  = findViewById(R.id.onOff);


        Button train = (Button)findViewById(R.id.btn_train);
        train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent swap = new Intent(MainActivity.this, TrainActivity.class);
                startActivity(swap);
            }
        });
        Button recognize = (Button)findViewById(R.id.btn_recognize);
        recognize.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent swap = new Intent(MainActivity.this, RecognizeActivity.class);
                startActivity(swap);
            }
        });

        function.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                isOn = !isOn;
                if(isOn){
                    onOff.setChecked(true);
                    action.setText("Face Unlock is Activated");
                }else{
                    onOff.setChecked(false);
                    action.setText("Face Unlock is Deactivated");
                }
            }
        });
    }
}
