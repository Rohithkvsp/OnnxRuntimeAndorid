//
// Created by rohith on 7/18/18.
//


#include <jni.h>
#include <string>

#include <android/bitmap.h>
#include "onnxruntime_inference.h"
#include "imageresizer.h"

#include "logs.h"



extern "C"
{

JNIEXPORT jlong JNICALL
Java_com_play_onnxruntime_Inference_newSelf(JNIEnv *env, jobject,jstring model_path, jstring label_file_path, jint img_height, jint img_width) {


    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING,"test");
    const char *model_path_ch = env->GetStringUTFChars(model_path, 0);
    const char *label_file_path_ch = env->GetStringUTFChars(label_file_path, 0);


    Inference *self = new Inference(environment, model_path_ch, label_file_path_ch, img_height, img_width);
    return (jlong) self;


}

JNIEXPORT void JNICALL
Java_com_play_onnxruntime_Inference_deleteSelf(JNIEnv *env, jobject,
                                                          jlong selfAddr) {
    if (selfAddr != 0) {
        Inference *self = (Inference *) selfAddr;
        LOGE("deleted c++ object");
        delete self;

    }
}

JNIEXPORT jstring JNICALL
Java_com_play_onnxruntime_Inference_run(JNIEnv *env, jobject, jlong selfAddr,
                                                   jobject inputbitmap) {
    if (selfAddr != 0) {
        AndroidBitmapInfo info;
        Inference *self = (Inference *) selfAddr;

        uint8_t *inputpixel;
        int ret;
        if ((ret = AndroidBitmap_getInfo(env, inputbitmap, &info)) < 0) {
            LOGE("Input AndroidBitmap_getInfo() failed ! error=%d", ret);
            return env->NewStringUTF("none");
        }

        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGE("InputBitmap format is not RGBA_8888 !");
            return env->NewStringUTF("none");
        }
        if ((ret = AndroidBitmap_lockPixels(env, inputbitmap, (void **) &inputpixel)) < 0) {
            LOGE("Input AndroidBitmap_lockPxels() failed ! error=%d", ret);
        }

        LOGD("bitmap width %d , bitmap heigth %d, bitmap stride %d", info.width, info.height, info.stride);


        AndroidBitmap_unlockPixels(env, inputbitmap);

        self->run(inputpixel);


        jstring label = env->NewStringUTF(self->getPredictedlabels().c_str());
        return label;

    }


    return env->NewStringUTF("None");;

}

}