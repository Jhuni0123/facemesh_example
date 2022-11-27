#include <gst/gst.h>
#include <nnstreamer/nnstreamer_plugin_api_decoder.h>
#include <nnstreamer/nnstreamer_plugin_api_util.h>
#include <nnstreamer/nnstreamer_plugin_api.h>
#include <nnstreamer/tensor_decoder_custom.h>
#include <nnstreamer/tensor_filter_custom.h>
#include <nnstreamer/tensor_filter_custom_easy.h>
#include <nnstreamer/tensor_typedef.h>
#include <nnstreamer/nnstreamer_util.h>
#include <math.h>
#include <stdlib.h>

#include "face_detect.c"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Macro for debug message.
 */
#define _print_log(...) if (DBG) g_message (__VA_ARGS__)

typedef struct
{
  gchar *model_path;

  guint tensor_width;
  guint tensor_height;

  guint i_width;
  guint i_height;
} LandmarkModelInfo;

/**
 * @brief Data structure for app.
 */
typedef struct
{
  BlazeFaceInfo detect_model;
  LandmarkModelInfo landmark_model;

  guint video_size;

  GstElement *pipeline; /**< gst pipeline for data stream */

  GMainLoop *loop; /**< main event loop */
  GstBus *bus; /**< gst bus for data pipeline */
} AppData;

gboolean
build_pipeline (AppData *app)
{
  GstElement *tee_source, *tee_cropinfo;
  GstPad *cropped_video_srcpad, *landmark_overray_srcpad;

  app->pipeline = gst_pipeline_new ("facemesh-pipeline");
  if (!app->pipeline) {
    g_printerr ("pipeline could not be created.\n");
    return FALSE;
  }

  /* Souece video */
  {
    GstElement *video_source, *convert, *filter, *crop, *scale;
    GstCaps *video_caps;

    video_source = gst_element_factory_make ("v4l2src", "video_source");
    convert = gst_element_factory_make ("videoconvert", "convert_source");
    filter = gst_element_factory_make ("capsfilter", "filter1");
    crop = gst_element_factory_make ("aspectratiocrop", "crop_source");
    scale = gst_element_factory_make ("videoscale", "scale_source");
    tee_source = gst_element_factory_make ("tee", "tee_source");

    if (!video_source || !convert || !filter || !crop || !tee_source) {
      g_printerr ("[SOUECE] Not all elements could be created.\n");
      return FALSE;
    }

    g_object_set (crop, "aspect-ratio", 1, 1, NULL);

    video_caps = gst_caps_new_simple ("video/x-raw",
       "format", G_TYPE_STRING, "RGB",
       "width", G_TYPE_INT, app->video_size,
       "height", G_TYPE_INT, app->video_size,
       NULL);
    g_object_set (G_OBJECT (filter), "caps", video_caps, NULL);
    gst_caps_unref (video_caps);

    gst_bin_add_many (GST_BIN (app->pipeline), video_source, convert, crop, scale, filter, tee_source, NULL);
    if (!gst_element_link_many (video_source, convert, crop, scale, filter, tee_source, NULL)) {
      g_printerr ("[SOURCE] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }

  }

  /* Face Detection to Crop */
  {
    GstElement *queue;
    GstElement *scale, *filter, *tconv, *ttransform, *tfilter_detect, *tfilter_cropinfo;
    GstCaps *scale_caps;
    GstPad *tee_pad, *queue_pad;
    BlazeFaceInfo *info;

    info = &app->detect_model;

    queue = gst_element_factory_make ("queue", "queue_detect");
    scale = gst_element_factory_make ("videoscale", "scale_detect");
    filter = gst_element_factory_make ("capsfilter", "filter_detect");
    tconv = gst_element_factory_make ("tensor_converter", "tconv_detect");
    ttransform = gst_element_factory_make ("tensor_transform", "ttransform_detect");
    tfilter_detect = gst_element_factory_make ("tensor_filter", "tfilter_detect");
    tfilter_cropinfo = gst_element_factory_make ("tensor_filter", "filter_cropinfo");
    tee_cropinfo = gst_element_factory_make ("tee", "tee_cropinfo");

    if (!queue || !scale || !filter || !tconv || !ttransform || !tfilter_detect || !tfilter_cropinfo || !tee_cropinfo) {
      g_printerr ("[DETECT] Not all elements could be created.\n");
      return FALSE;
    }

    scale_caps = gst_caps_new_simple ("video/x-raw",
       "format", G_TYPE_STRING, "RGB",
       "framerate", GST_TYPE_FRACTION, 30, 1,
       "width", G_TYPE_INT, info->tensor_width,
       "height", G_TYPE_INT, info->tensor_height,
       NULL);
    g_object_set (G_OBJECT (filter), "caps", scale_caps, NULL);
    gst_caps_unref (scale_caps);

    g_object_set (ttransform, "mode", 2 /* GTT_ARITHMETIC */, "option", "typecast:float32,add:-127.5,div:127.5", NULL);
    g_object_set (tfilter_detect, "framework", "tensorflow-lite", "model", app->detect_model.model_path, NULL);
    g_object_set (tfilter_cropinfo, "framework", "custom-easy", "model", "detection_to_cropinfo", NULL);

    gst_bin_add_many (GST_BIN (app->pipeline), 
        queue, scale, filter, tconv, ttransform, tfilter_detect, tfilter_cropinfo, tee_cropinfo, NULL);

    if (!gst_element_link_many (queue, scale, filter, tconv, ttransform, tfilter_detect, tfilter_cropinfo, tee_cropinfo, NULL)) {
      g_printerr ("[DETECT] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }

    tee_pad = gst_element_request_pad_simple (tee_source, "src_%u");
    _print_log ("[DETECT] Obtained request pad %s\n", gst_pad_get_name (tee_pad));
    queue_pad = gst_element_get_static_pad (queue, "sink");

    if (gst_pad_link (tee_pad, queue_pad) != GST_PAD_LINK_OK) {
      g_printerr ("[DETECT] Tee could not be linked\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    gst_object_unref (queue_pad);
  }

  /* Crop video */
  {
    GstElement *queue_cropinfo, *queue, *tconv, *tcrop;
    GstPad *tee_pad, *queue_pad;
    GstPad *tee_cropinfo_pad, *queue_cropinfo_pad;

    queue_cropinfo = gst_element_factory_make ("queue", "queue_cropinfo1");
    queue = gst_element_factory_make ("queue", "queue_crop");
    tconv = gst_element_factory_make ("tensor_converter", "tconv_crop");
    tcrop = gst_element_factory_make ("tensor_crop", "tcrop");

    if (!queue || !tconv || !tcrop) {
      g_printerr ("[CROP] Not all elements could be created.\n");
      return FALSE;
    }

    gst_bin_add_many (GST_BIN (app->pipeline), queue_cropinfo, queue, tconv, tcrop, NULL);

    if (!gst_element_link_many (queue, tconv, NULL)
        || !gst_element_link_pads (tconv, "src", tcrop, "raw")
        || !gst_element_link_pads (queue_cropinfo, "src", tcrop, "info")) {
      g_printerr ("[CROP] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }

    tee_pad = gst_element_request_pad_simple (tee_source, "src_%u");
    _print_log ("[CROP] Obtained request pad %s\n", gst_pad_get_name (tee_pad));
    queue_pad = gst_element_get_static_pad (queue, "sink");

    tee_cropinfo_pad = gst_element_request_pad_simple (tee_cropinfo, "src_%u");
    _print_log ("[CROP_INFO] Obtained request pad %s\n", gst_pad_get_name (tee_cropinfo_pad));
    queue_cropinfo_pad = gst_element_get_static_pad (queue_cropinfo, "sink");

    if (gst_pad_link (tee_pad, queue_pad) != GST_PAD_LINK_OK
        || gst_pad_link (tee_cropinfo_pad, queue_cropinfo_pad) != GST_PAD_LINK_OK) {
      g_printerr ("[CROP] Tee could not be linked\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    gst_object_unref (queue_pad);

    cropped_video_srcpad = gst_element_get_static_pad (tcrop, "src");
  }

  /* Cropped video to videosink */
  /*
  {
    GstElement *tdec_flexible, *convert_crop, *video_sink_crop;
    GstPad *cropped_video_sinkpad;
    GstCaps *cropped_video_caps;

    tdec_flexible = gst_element_factory_make ("tensor_decoder", "tdec_flexible");
    convert_crop = gst_element_factory_make ("videoconvert", "convert_crop");
    video_sink_crop = gst_element_factory_make ("autovideosink", "video_sink_crop");

    if (!tdec_flexible || !convert_crop || !video_sink_crop) {
      g_printerr ("[CROPPED VIDEO] Not all elements could be created.\n");
      return FALSE;
    }

    g_object_set (tdec_flexible, "mode", "custom-code", "option1", "flexible_to_video", NULL);

    gst_bin_add_many (GST_BIN (app->pipeline), tdec_flexible, convert_crop, video_sink_crop, NULL);

    cropped_video_sinkpad = gst_element_get_static_pad (tdec_flexible, "sink");
    cropped_video_caps = gst_caps_from_string ("video/x-raw,format=RGB,width=192,height=192,framerate=30/1");
    if (gst_pad_link (cropped_video_srcpad, cropped_video_sinkpad) != GST_PAD_LINK_OK
        || !gst_element_link_filtered (tdec_flexible, convert_crop, cropped_video_caps)
        || !gst_element_link_many (convert_crop, video_sink_crop, NULL)
    ) {
      g_printerr ("[CROPPED VIDEO] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    gst_caps_unref (cropped_video_caps);
  }
  */

  /* Face Landmark */
  {
    GstElement *queue, *tdec_flexible, *tconv, *ttransform, *tfilter_landmark, *tdec_landmark;
    GstPad *cropped_video_sinkpad;
    GstCaps *cropped_video_caps;
    LandmarkModelInfo *info;
    gchar *input_size, *output_size;

    info = &app->landmark_model;

    queue = gst_element_factory_make ("queue", "queue_landmark");
    tdec_flexible = gst_element_factory_make ("tensor_decoder", "tdec_flexible");
    tconv = gst_element_factory_make ("tensor_converter", "tconv_landmark");
    ttransform = gst_element_factory_make ("tensor_transform", "ttransform_landmark");
    tfilter_landmark = gst_element_factory_make ("tensor_filter", "tfilter_landmark");
    tdec_landmark = gst_element_factory_make ("tensor_decoder", "tdec_landmark");

    if (!queue || !tdec_flexible || !tconv || !ttransform || !tfilter_landmark || !tdec_landmark) {
      g_printerr ("[LANDMARK] Not all elements could be created.\n");
      return FALSE;
    }

    g_object_set (tdec_flexible, "mode", "custom-code", "option1", "flexible_to_video", NULL);
    g_object_set (ttransform, "mode", 2 /* GTT_ARITHMETIC */, "option", "typecast:float32,add:-127.5,div:127.5", NULL);
    g_object_set (tfilter_landmark, "framework", "tensorflow-lite", "model", app->landmark_model.model_path, NULL);

    input_size = g_strdup_printf ("%d:%d", info->tensor_width, info->tensor_height);
    output_size = g_strdup_printf ("%d:%d", app->video_size, app->video_size);
    g_object_set (tdec_landmark, "mode", "face_mesh", "option1", "mediapipe-face-mesh", "option2", output_size, "option3", input_size, NULL);
    g_free (input_size);
    g_free (output_size);

    gst_bin_add_many (GST_BIN (app->pipeline),
        queue, tdec_flexible, tconv, ttransform, tfilter_landmark, tdec_landmark, NULL);

    cropped_video_sinkpad = gst_element_get_static_pad (queue, "sink");

    cropped_video_caps = gst_caps_new_simple ("video/x-raw",
       "format", G_TYPE_STRING, "RGB",
       "framerate", GST_TYPE_FRACTION, 30, 1,
       "width", G_TYPE_INT, info->tensor_width,
       "height", G_TYPE_INT, info->tensor_height,
       NULL);
    if (gst_pad_link (cropped_video_srcpad, cropped_video_sinkpad) != GST_PAD_LINK_OK
        || !gst_element_link (queue, tdec_flexible)
        || !gst_element_link_filtered (tdec_flexible, tconv, cropped_video_caps)
        || !gst_element_link_many (tconv, ttransform, tfilter_landmark, tdec_landmark, NULL))
    {
      g_printerr ("[LANDMARK] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    
    gst_caps_unref (cropped_video_caps);

    landmark_overray_srcpad = gst_element_get_static_pad (tdec_landmark, "src");
  }

  /* Result video */
  {
    GstElement *queue, *compositor, *convert, *video_sink, *queue_cropinfo, *crop_scale;
    GstPad *tee_pad, *queue_pad, *compositor_overray_pad, *compositor_video_pad, *overray_pad, *overray_raw_pad;
    GstPad *tee_cropinfo_pad, *queue_cropinfo_pad;

    queue = gst_element_factory_make ("queue", "queue_result");
    queue_cropinfo = gst_element_factory_make ("queue", "queue_cropinfo2");
    compositor = gst_element_factory_make ("compositor", "compositor");
    convert = gst_element_factory_make ("videoconvert", "convert_result");
    video_sink = gst_element_factory_make ("autovideosink", "video_sink");
    crop_scale = gst_element_factory_make ("crop_scale", "crop_scale");

    if (!queue || !compositor || !convert || !video_sink || !queue_cropinfo || !crop_scale) {
      g_printerr ("[RESULT] Not all elements could be created.\n");
      return FALSE;
    }

    gst_bin_add_many (GST_BIN (app->pipeline), queue, compositor, convert, video_sink, queue_cropinfo, crop_scale, NULL);
    
    overray_raw_pad = gst_element_get_static_pad (crop_scale, "raw");
    if (!gst_element_link_many (compositor, convert, video_sink, NULL)
        || !gst_element_link_pads (queue_cropinfo, "src", crop_scale, "info")
        || gst_pad_link (landmark_overray_srcpad, overray_raw_pad) != GST_PAD_LINK_OK) {
      g_printerr ("[RESULT] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }

    compositor_overray_pad = gst_element_request_pad_simple (compositor, "sink_%u");
    _print_log ("[RESULT] Obtained request pad %s\n", gst_pad_get_name (compositor_overray_pad));
    g_object_set (compositor_overray_pad, "zorder", 2, NULL);

    compositor_video_pad = gst_element_request_pad_simple (compositor, "sink_%u");
    _print_log ("[RESULT] Obtained request pad %s\n", gst_pad_get_name (compositor_video_pad));
    g_object_set (compositor_video_pad, "zorder", 1, NULL);
    queue_pad = gst_element_get_static_pad (queue, "src");

    overray_pad = gst_element_get_static_pad (crop_scale, "src");
    if (gst_pad_link (overray_pad, compositor_overray_pad) != GST_PAD_LINK_OK
        || gst_pad_link (queue_pad, compositor_video_pad) != GST_PAD_LINK_OK) {
      g_printerr ("[RESULT] Compositor could not be linked\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    gst_object_unref (queue_pad);

    tee_pad = gst_element_request_pad_simple (tee_source, "src_%u");
    _print_log ("[RESULT] Obtained request pad %s\n", gst_pad_get_name (tee_pad));
    queue_pad = gst_element_get_static_pad (queue, "sink");

    tee_cropinfo_pad = gst_element_request_pad_simple (tee_cropinfo, "src_%u");
    _print_log ("[CROP_INFO] Obtained request pad %s\n", gst_pad_get_name (tee_cropinfo_pad));
    queue_cropinfo_pad = gst_element_get_static_pad (queue_cropinfo, "sink");

    if (gst_pad_link (tee_pad, queue_pad) != GST_PAD_LINK_OK
        || gst_pad_link (tee_cropinfo_pad, queue_cropinfo_pad) != GST_PAD_LINK_OK) {
      g_printerr ("[RESULT] Tee could not be linked\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    gst_object_unref (queue_pad);
  }

  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (app->pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");
  return TRUE;
}

static void
margin_object(detectedObject *orig, detectedObject *margined, gfloat margin_rate, guint video_size)
{
  gint height = orig->height;
  gint width = orig->width;
  gint orig_size = MAX(height, width);
  gint margin = orig_size * margin_rate;
  gint margined_size = MIN(orig_size + margin * 2, video_size);
  gint x = MIN(MAX(orig->x - margin, 0), video_size - margined_size);
  gint y = MIN(MAX(orig->y - margin, 0), video_size - margined_size);
  margined->x = x;
  margined->y = y;
  margined->width = margined_size;
  margined->height = margined_size;
}

/**
 * @brief In-Code Test Function for custom-easy filter
 */
static int
cef_func_detection_to_cropinfo (void *private_data, const GstTensorFilterProperties *prop,
    const GstTensorMemory *in, GstTensorMemory *out)
{
  AppData *app = private_data;
  BlazeFaceInfo *info = &app->detect_model;
  float *raw_boxes = in[0].data;
  float *raw_scores = in[1].data;
  GArray *results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), 100);
  guint *info_data = out[0].data;

  for (guint i = 0; i < info->num_boxes; i++) {
    detectedObject object = {.valid = FALSE, .class_id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .prob = 0};

    get_detected_object_i (i, raw_boxes, raw_scores, info, &object);

    if (object.valid) {
      g_array_append_val (results, object);
    }
  }

  nms (results, info->iou_thresh);

  if (results->len == 0) {
    //_print_log ("no detected object");
    info_data[0] = 0U;
    info_data[1] = 0U;
    info_data[2] = info->i_width;
    info_data[3] = info->i_height;
  } else {
    detectedObject *object = &g_array_index (results, detectedObject, 0);
    detectedObject margined;

    margin_object (object, &margined, 0.25, app->video_size);
    //_print_log ("detected: %d %d %d %d = %d", object->x, object->y, object->height, object->width, object->height * object->width * 3);
    //_print_log ("detected: %d %d %d %d = %d", margined.x, margined.y, margined.height, margined.width, margined.height * margined.width * 3);

    info_data[0] = margined.x;
    info_data[1] = margined.y;
    info_data[2] = margined.width;
    info_data[3] = margined.height;
  }
  return 0;
}

static size_t
_get_video_xraw_RGB_bufsize (size_t width, size_t height)
{
  return (size_t)((3 * width - 1) / 4 + 1) * 4 * height;
}

int flexible_tensor_to_video (const GstTensorMemory *input, const GstTensorsConfig *config, void *data, GstBuffer *out_buf) {
  AppData *app = data;
  GstMapInfo out_info;
  GstMemory *out_mem;
  GstTensorMetaInfo meta;
  gsize hsize, dsize, esize;
  const GstTensorMemory *tmem = &input[0];
  gboolean need_alloc;
  guint width, height;

  width = app->landmark_model.tensor_width;
  height = app->landmark_model.tensor_height;

  g_assert (gst_tensors_config_is_flexible(config));
  g_assert (config->info.num_tensors >= 1);

  if (!gst_tensor_meta_info_parse_header (&meta, (gpointer) tmem->data)) {
    GST_ERROR ("Invalid tensor meta info.");
    return GST_FLOW_ERROR;
  }

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  dsize = gst_tensor_meta_info_get_data_size (&meta);
  esize = gst_tensor_get_element_size (meta.type);

  //_print_log ("size: %zd %zd %zd", hsize, dsize, esize);
  g_assert (1 == esize);

  if (hsize + dsize != tmem->size) {
    GST_ERROR ("Invalid tensor meta info.");
    return GST_FLOW_ERROR;
  }

  const uint32_t *dim = &meta.dimension[0];
  g_assert (3 == dim[0]);
  g_assert (dim[1] == dim[2]);

  size_t size = _get_video_xraw_RGB_bufsize (width, height);

  need_alloc = (gst_buffer_get_size (out_buf) == 0);

  if (need_alloc) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (out_buf)) {
      gst_buffer_set_size (out_buf, size);
    }
    out_mem = gst_buffer_get_all_memory (out_buf);
  }

  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    gst_memory_unref (out_mem);
    return GST_FLOW_ERROR;
  }
  
  //memset (out_info.data, 0xFF0000FF, size);
  /* neareast-neighbor */
  int h, w;
  uint8_t *ptr = (uint8_t *)out_info.data;
  uint8_t *inp = (uint8_t *)tmem->data + hsize;
  for (h = 0; h < height; h++) {
    int h_inp = (int)((float)dim[2] / height * h);
    uint8_t *row_inp = inp + dim[0] * dim[1] * h_inp;
    uint8_t *row_ptr = ptr;
    for (w = 0; w < width; w++) {
      int w_inp = (int)((float)dim[1] / width * w);
      uint8_t *pix_inp = row_inp + dim[0] * w_inp;
      memcpy (row_ptr, pix_inp, 3);
      row_ptr += 3;
    }
    ptr += ((3 * width - 1) / 4 + 1) * 4;
  }

  gst_memory_unmap (out_mem, &out_info);

  if (need_alloc) {
    gst_buffer_append_memory (out_buf, out_mem);
  } else {
    gst_memory_unref (out_mem);
  }

  return GST_FLOW_OK;
}

static gboolean
init_blazeface (BlazeFaceInfo *info, const gchar *path, guint video_size)
{
  const gchar detect_model[] = "face_detection_short_range.tflite";
  const gchar detect_box_prior[] = "box_prior_face_detection_short_range.txt";
  const guint num_boxs = BLAZEFACE_SHORT_RANGE_NUM_BOXS;

  info->model_path = g_strdup_printf ("%s/%s", path, detect_model);
  info->anchors_path = g_strdup_printf ("%s/%s", path, detect_box_prior);
  info->num_boxes = num_boxs;
  info->x_scale = 128;
  info->y_scale = 128;
  info->h_scale = 128;
  info->w_scale = 128;
  info->min_score_thresh = 0.5f;
  info->iou_thresh = 0.3f;
  info->tensor_width = 128;
  info->tensor_height = 128;

  info->i_width = video_size;
  info->i_height = video_size;

  if (!g_file_test (info->model_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("cannot find tflite model [%s]", info->model_path);
    return FALSE;
  }

  if (!g_file_test (info->anchors_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("cannot find tflite box_prior [%s]", info->anchors_path);
    return FALSE;
  }

  blazeface_load_anchors (info);

  return TRUE;
}

static gboolean
init_landmark_model (LandmarkModelInfo *info, const gchar *path, guint video_size)
{
  const gchar landmark_model[] = "face_landmark.tflite";

  info->model_path = g_strdup_printf ("%s/%s", path, landmark_model);
  info->tensor_width = 192;
  info->tensor_height = 192;
  info->i_width = video_size;
  info->i_height = video_size;

  if (!g_file_test (info->model_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("cannot find tflite model [%s]", info->model_path);
    return FALSE;
  }

  return TRUE;
}

gboolean 
init_app (AppData *app)
{
  const gchar resource_path[] = "./res";

  app->video_size = 720;
  init_blazeface (&app->detect_model, resource_path, app->video_size);
  init_landmark_model (&app->landmark_model, resource_path, app->video_size);


  app->loop = g_main_loop_new (NULL, FALSE);

  GstTensorsInfo info_in;
  GstTensorsInfo info_out;

  /* register custom crop_info filter */
  gst_tensors_info_init (&info_in);
  gst_tensors_info_init (&info_out);
  info_in.num_tensors = 2U;
  info_in.info[0].name = NULL;
  info_in.info[0].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("16:896", info_in.info[0].dimension);
  info_in.info[1].name = NULL;
  info_in.info[1].type = _NNS_FLOAT32;
  gst_tensor_parse_dimension ("896", info_in.info[1].dimension);

  info_out.num_tensors = 1U;
  info_out.info[0].name = NULL;
  info_out.info[0].type = _NNS_UINT32;
  gst_tensor_parse_dimension ("4:1", info_out.info[0].dimension);
  NNS_custom_easy_register ("detection_to_cropinfo", cef_func_detection_to_cropinfo, app, &info_in, &info_out);

  /* register custom flexible tensor to video decoder */
  nnstreamer_decoder_custom_register ("flexible_to_video", flexible_tensor_to_video, app);

  if (!build_pipeline (app)) {
    return FALSE;
  }

  return TRUE;
}

static void
message_cb (GstBus *bus, GstMessage *msg, AppData *app)
{
  GError *err;
  gchar *debug_info;

  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_ERROR:
      gst_message_parse_error (msg, &err, &debug_info);
      g_printerr ("Error received from element %s: %s\n", GST_OBJECT_NAME (msg->src), err->message);
      g_printerr ("Debug information: %s\n", debug_info ? debug_info : "none");
      g_clear_error (&err);
      g_free (debug_info);
      g_main_loop_quit (app->loop);
      break;
    default:
      // g_printerr ("msg from %s:  %s\n", GST_OBJECT_NAME (msg->src), gst_message_type_get_name (GST_MESSAGE_TYPE (msg)));
      break;
  }
}

int
main (int argc, char *argv[])
{
  AppData app;

  GstBus *bus;

  /* Initialize GStreamer */
  gst_init (&argc, &argv);

  /* Build the pipeline */
  if (!init_app (&app)) {
    return -1;
  }

  /* connect message handler */
  bus = gst_element_get_bus (app.pipeline);
  gst_bus_add_signal_watch (bus);
  g_signal_connect (G_OBJECT (bus), "message", (GCallback) message_cb, &app);

  /* Start pipeline */
  gst_element_set_state (app.pipeline, GST_STATE_PLAYING);
  g_main_loop_run (app.loop);

  /* Free resources */
  gst_object_unref (bus);
  gst_element_set_state (app.pipeline, GST_STATE_NULL);
  gst_object_unref (app.pipeline);
  g_main_loop_unref (app.loop);
  return 0;
}
