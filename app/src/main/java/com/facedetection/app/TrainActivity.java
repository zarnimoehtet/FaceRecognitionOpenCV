package com.facedetection.app;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Dialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.drawable.ColorDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.annotation.Nullable;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.text.InputType;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.face.Face;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.util.FloatMath;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static org.opencv.core.Core.NORM_L2;
import static org.opencv.core.Core.norm;
import static org.opencv.objdetect.Objdetect.CASCADE_SCALE_IMAGE;


public class TrainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static String TAG = TrainActivity.class.getSimpleName();
    private CameraBridgeViewBase openCVCamera;
    private Mat rgba, gray;
    private CascadeClassifier classifier;
    private MatOfRect faces;
    private static final int PERMS_REQUEST_CODE = 123;
    private ArrayList<Mat> images;
    private ArrayList<String> imagesLabels;
    private Storage local;
    private String[] uniqueLabels;
    private Boolean isUseGalleryImage = false;
    ArrayList<Mat> tempList;
    FaceRecognizer recognize;
    public static final int PICK_IMAGE = 1;

    int count = 0;
    Mat temp ;

    private boolean trainfaces() {
        if (images.isEmpty())
            return false;

        List<Mat> imagesMatrix = new ArrayList<>();
        for (int i = 0; i < images.size(); i++)
            imagesMatrix.add(images.get(i));
        Set<String> uniqueLabelsSet = new HashSet<>(imagesLabels); // Get all unique labels
        uniqueLabels = uniqueLabelsSet.toArray(new String[uniqueLabelsSet.size()]); // Convert to String array, so we can read the values from the indices

        int[] classesNumbers = new int[uniqueLabels.length];
        for (int i = 0; i < classesNumbers.length; i++)
            classesNumbers[i] = i + 1; // Create incrementing list for each unique label starting at 1
        int[] classes = new int[imagesLabels.size()];
        for (int i = 0; i < imagesLabels.size(); i++) {
            String label = imagesLabels.get(i);
            Log.i("Label ", label);
            for (int j = 0; j < uniqueLabels.length - images.size(); j++) {
                Log.i("Unique", uniqueLabels[j]);
                if (label.equals(uniqueLabels[j])) {
                    Log.i("ClassesNumber", classesNumbers[j] + "");
                    classes[i] = classesNumbers[j]; // Insert corresponding number
                    break;
                }
            }
        }
        Mat vectorClasses = new Mat(classes.length, 1, CvType.CV_32SC1); // CV_32S == int
        vectorClasses.put(0, 0, classes); // Copy int array into a vector
        Log.i("ClassesLength", classes.length + "");
        Log.i("VectorClassesLength", vectorClasses.cols() + " " + vectorClasses.rows());
        Log.i("imageMatrixLength", imagesMatrix.size() + "");

        recognize = LBPHFaceRecognizer.create(3, 8, 8, 8, 200);
        try {
            recognize.train(imagesMatrix, vectorClasses);
        } catch (Exception e) {
            Log.i(TAG, e.toString());
        }

        if (SaveImage())
            return true;

        return false;
    }

    public void showLabelsDialog() {
        Set<String> uniqueLabelsSet = new HashSet<>(imagesLabels); // Get all unique labels
        if (!uniqueLabelsSet.isEmpty()) {
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("Select Name");
            builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    dialog.dismiss();
                    images.remove(images.size() - 1);
                }
            });
            builder.setCancelable(false); // Prevent the user from closing the dialog

            String[] uniqueLabels = uniqueLabelsSet.toArray(new String[uniqueLabelsSet.size()]); // Convert to String array for ArrayAdapter
            Arrays.sort(uniqueLabels); // Sort labels alphabetically
            final ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, uniqueLabels) {
                @Override
                public View getView(int position, View convertView, ViewGroup parent) {
                    TextView textView = (TextView) super.getView(position, convertView, parent);
                    if (getResources().getBoolean(R.bool.isTablet))
                        textView.setTextSize(20); // Make text slightly bigger on tablets compared to phones
                    else
                        textView.setTextSize(18); // Increase text size a little bit
                    return textView;
                }
            };
            ListView mListView = new ListView(this);
            mListView.setAdapter(arrayAdapter); // Set adapter, so the items actually show up
            builder.setView(mListView); // Set the ListView

            final AlertDialog dialog = builder.show(); // Show dialog and store in final variable, so it can be dismissed by the ListView

            mListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                @Override
                public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                    dialog.dismiss();
                    addLabel(arrayAdapter.getItem(position));

                    Log.i(TAG, "Labels Size " + imagesLabels.size() + "");
                    Log.i(TAG, "ImageSize " + images.size());
                }
            });
        } else {
            showEnterLabelDialog();
        }

    }

    private void showEnterLabelDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Please enter your name:");

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(input);

        builder.setPositiveButton("Submit", null); // Set up positive button, but do not provide a listener, so we can check the string before dismissing the dialog
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
                images.remove(images.size() - 1);
            }
        });
        builder.setCancelable(false); // User has to input a name
        AlertDialog dialog = builder.create();

        // Source: http://stackoverflow.com/a/7636468/2175837
        dialog.setOnShowListener(new DialogInterface.OnShowListener() {
            @Override
            public void onShow(final DialogInterface dialog) {
                Button mButton = ((AlertDialog) dialog).getButton(AlertDialog.BUTTON_POSITIVE);
                mButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        String string = input.getText().toString().trim();
                        if (!string.isEmpty()) { // Make sure the input is valid
                            // If input is valid, dismiss the dialog and add the label to the array
                            dialog.dismiss();
                            addLabel(string);
                        }
                    }
                });
            }
        });
        // Show keyboard, so the user can start typing straight away
        dialog.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE);

        dialog.show();
    }

    private void addLabel(String string) {
        String label = string.substring(0, 1).toUpperCase(Locale.US) + string.substring(1).trim().toLowerCase(Locale.US); // Make sure that the name is always uppercase and rest is lowercase
        imagesLabels.add(label); // Add label to list of labels
        Log.i(TAG, "Label: " + label);
       // trainfaces();

    }

    public boolean SaveImage() {
        File path = new File(Environment.getExternalStorageDirectory(), "TrainedData");
        path.mkdirs();
        String filename = "lbph_trained_data.xml";
        File file = new File(path, filename);
        recognize.save(file.toString());
        if (file.exists())
            return true;
        return false;
    }

    public void cropedImages(Mat mat) {
        Rect rect_Crop = null;
        for (Rect face : faces.toArray()) {
            rect_Crop = new Rect(face.x, face.y, 40, 40);
        }
        Mat croped = new Mat(mat, rect_Crop);
        //calculate distance two image mat using c++ lib
        Log.i(TAG, "MAT Size " + croped.rows() + " " + croped.cols());
        // ShowBitmapDialog(croped);

        if (count > 0) {
            double dist = Core.norm(temp, croped, NORM_L2);
            temp = croped;
            Toast.makeText(this, "Value " + dist, Toast.LENGTH_SHORT).show();
        }else{
            temp = new Mat();
            temp = croped;
        }

        trainfaces();


        images.add(croped);
        Log.i(TAG, "Image Size " + images.size());
        Log.i(TAG, "Lable Size " + imagesLabels.size());
      //  trainfaces();

//        ShowBitmapDialog(images.get(count));
        count++;

    }

    void ShowBitmapDialog(Mat m) {
        Log.i("Data", m.rows() + " " + m.cols());
        Bitmap bmp = null;
        Mat tmp = new Mat();
        try {
            Imgproc.cvtColor(m, tmp, Imgproc.COLOR_RGB2BGRA);
            bmp = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(tmp, bmp);
        } catch (CvException e) {
            Log.d("MtoB", e.getMessage());
        }
        Dialog builder = new Dialog(this);
        builder.requestWindowFeature(Window.FEATURE_NO_TITLE);
        builder.getWindow().setBackgroundDrawable(
                new ColorDrawable(android.graphics.Color.TRANSPARENT));
        builder.setOnDismissListener(new DialogInterface.OnDismissListener() {
            @Override
            public void onDismiss(DialogInterface dialogInterface) {
                //nothing;
            }
        });

        ImageView imageView = new ImageView(this);
        imageView.setImageBitmap(bmp);
        builder.addContentView(imageView, new RelativeLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));
        builder.show();
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.train_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // Stetho.initializeWithDefaults(this);

        if (hasPermissions()) {
            Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT).show();
            Log.i(TAG, "Permission Granted Before");
        } else {
            requestPerms();
        }

        openCVCamera = (CameraBridgeViewBase) findViewById(R.id.java_camera_view);
        openCVCamera.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        openCVCamera.setVisibility(SurfaceView.VISIBLE);
        openCVCamera.setCvCameraViewListener(this);
        local = new Storage(this);
        Button detect = (Button) findViewById(R.id.take_picture_button);

        detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (gray.total() == 0)
                    Toast.makeText(getApplicationContext(), "Can't Detect Faces", Toast.LENGTH_SHORT).show();
                classifier.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, new Size(30, 30));
                if (!faces.empty()) {
                    if (faces.toArray().length > 1)
                        Toast.makeText(getApplicationContext(), "Multiple Faces Are not allowed", Toast.LENGTH_SHORT).show();
                    else {
                        if (gray.total() == 0) {
                            Log.i(TAG, "Empty gray image");
                            return;
                        }
                        cropedImages(gray);
                        isUseGalleryImage = false;
                        showLabelsDialog();
                        Toast.makeText(getApplicationContext(), "Face Detected", Toast.LENGTH_SHORT).show();
                    }
                } else
                    Toast.makeText(getApplicationContext(), "Unknown Face", Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE) {
            Uri selectedImageUri = data.getData();
            if (null != selectedImageUri) {
                try {

                    final InputStream imageStream = getContentResolver().openInputStream(selectedImageUri);
                    final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                    // final Bitmap grayImage = getGrayScaleBitmap(selectedImage);

                    Mat src = new Mat();
                    Utils.bitmapToMat(selectedImage, src);
                    if (src.total() == 0)

                        Toast.makeText(getApplicationContext(), "Can't Detect Faces", Toast.LENGTH_SHORT).show();
                    classifier.detectMultiScale(src, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, new Size(30, 30));
                    if (!faces.empty()) {
                        if (faces.toArray().length > 1) {

                            Toast.makeText(getApplicationContext(), "Multiple Faces Are not allowed", Toast.LENGTH_SHORT).show();
                        } else {
                            if (src.total() == 0) {
                                Log.i(TAG, "Empty gray image");
                                return;
                            }


                            cropedImages(src);
                            isUseGalleryImage = true;
                            showLabelsDialog();

                            Toast.makeText(getApplicationContext(), "Face Detected", Toast.LENGTH_SHORT).show();
                        }
                    } else {

                        Toast.makeText(getApplicationContext(), "Unknown Face", Toast.LENGTH_SHORT).show();
                    }

                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }

        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // R.menu.mymenu is a reference to an xml file named mymenu.xml which should be inside your res/menu directory.
        // If you don't have res/menu, just create a directory named "menu" inside res
        getMenuInflater().inflate(R.menu.menu, menu);
        return super.onCreateOptionsMenu(menu);
    }

    // handle button activities
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if (id == R.id.mybutton) {
          startActivity(new Intent(TrainActivity.this,LocalImageTrain.class));
        }
        return super.onOptionsItemSelected(item);
    }

    private Bitmap getGrayScaleBitmap(Bitmap original) {
        // You have to make the Bitmap mutable when changing the config because there will be a crash
        // That only mutable Bitmap's should be allowed to change config.
        Bitmap bmp = original.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap bmpGrayscale = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0f);
        ColorFilter colorMatrixFilter = new ColorMatrixColorFilter(colorMatrix);
        paint.setColorFilter(colorMatrixFilter);
        canvas.drawBitmap(bmp, 0F, 0F, paint);
        return bmpGrayscale;
    }

    @SuppressLint("WrongConstant")
    private boolean hasPermissions() {
        int res = 0;
        //string array of permissions,
        String[] permissions = new String[]{Manifest.permission.CAMERA};

        for (String perms : permissions) {
            res = checkCallingOrSelfPermission(perms);
            if (!(res == PackageManager.PERMISSION_GRANTED)) {
                return false;
            }
        }
        return true;
    }

    private void requestPerms() {
        String[] permissions = new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE};
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(permissions, PERMS_REQUEST_CODE);

        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        boolean allowed = true;
        switch (requestCode) {
            case PERMS_REQUEST_CODE:
                for (int res : grantResults) {
                    // if user granted all permissions.
                    allowed = allowed && (res == PackageManager.PERMISSION_GRANTED);
                }
                break;
            default:
                // if user not granted permissions.
                allowed = false;
                break;
        }
        if (allowed) {
            //user granted all permissions we can perform our task.
            Log.i(TAG, "Permission has been added");
        } else {
            // we will give warning to user that they haven't granted permissions.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) || shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE) ||
                        shouldShowRequestPermissionRationale(Manifest.permission.READ_EXTERNAL_STORAGE)) {
                    Toast.makeText(this, "Permission Denied.", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    private BaseLoaderCallback callbackLoader = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS:
                    faces = new MatOfRect();
                    openCVCamera.enableView();

                    images = local.getListMat("images");
                    imagesLabels = local.getListString("imagesLabels");
                    Log.i(TAG, "onResume" + images.size());
                    break;
                default:
                    Log.i(TAG, "onResume Default" + images.size());
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onPause() {
        super.onPause();
        if (openCVCamera != null)
            openCVCamera.disableView();
        Log.i(TAG, "onPause" + images.size());


    }

    @Override
    protected void onStop() {
        super.onStop();
        if (images != null && imagesLabels != null) {
            local.putListMat("images", images);
            local.putListString("imagesLabels", imagesLabels);

//            if (trainfaces()) {
//                images.clear();
//                imagesLabels.clear();
//
//            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (openCVCamera != null)
            openCVCamera.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.i(TAG, "System Library Loaded Successfully");
            callbackLoader.onManagerConnected(BaseLoaderCallback.SUCCESS);
        } else {
            Log.i(TAG, "Unable To Load System Library");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, callbackLoader);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        rgba = new Mat();
        gray = new Mat();
        classifier = FileUtils.loadXMLS(this, "lbpcascade_frontalface_improved.xml");
    }

    @Override
    public void onCameraViewStopped() {
        rgba.release();
        gray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat mGrayTmp = inputFrame.gray();
        Mat mRgbaTmp = inputFrame.rgba();

        int orientation = openCVCamera.getScreenOrientation();
        if (openCVCamera.isEmulator()) // Treat emulators as a special case
            Core.flip(mRgbaTmp, mRgbaTmp, 1); // Flip along y-axis
        else {
            switch (orientation) { // RGB image
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    Core.flip(mRgbaTmp, mRgbaTmp, 0); // Flip along x-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    Core.flip(mRgbaTmp, mRgbaTmp, 1); // Flip along y-axis
                    break;
            }
            switch (orientation) { // Grayscale image
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                    Core.transpose(mGrayTmp, mGrayTmp); // Rotate image
                    Core.flip(mGrayTmp, mGrayTmp, -1); // Flip along both axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    Core.transpose(mGrayTmp, mGrayTmp); // Rotate image
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                    Core.flip(mGrayTmp, mGrayTmp, 1); // Flip along y-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    Core.flip(mGrayTmp, mGrayTmp, 0); // Flip along x-axis
                    break;
            }
        }
        gray = mGrayTmp;
        rgba = mRgbaTmp;
        Imgproc.resize(gray, gray, new Size(200, 200.0f / ((float) gray.width() / (float) gray.height())));
        return rgba;
    }
}
