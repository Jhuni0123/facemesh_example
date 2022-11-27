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

/**
 * @brief Sigmoid function
 */
#define sigmoid(x) \
    (1.f / (1.f + expf (- ((float)x))))

/**
 * @brief Function to print caps.
 */
static void
_parse_caps (GstCaps * caps)
{
  guint caps_size, i;

  g_return_if_fail (caps != NULL);

  caps_size = gst_caps_get_size (caps);

  for (i = 0; i < caps_size; i++) {
    GstStructure *structure = gst_caps_get_structure (caps, i);
    gchar *str = gst_structure_to_string (structure);

    _print_log ("[%d] %s", i, str);
    g_free (str);
  }
}

#define BLAZEFACE_SHORT_RANGE_NUM_BOXS  (896)
#define BLAZEFACE_NUM_COORD             (16)

typedef struct
{
  gchar *anchors_path;
  gchar *model_path;

  guint num_boxes;
#define ANCHOR_X_CENTER_IDX    (0)
#define ANCHOR_Y_CENTER_IDX    (1)
#define ANCHOR_WIDTH_IDX       (2)
#define ANCHOR_HEIGHT_IDX      (3)
#define ANCHOR_SIZE  (4)
#define DETECTION_MAX (896)
  gfloat anchors[DETECTION_MAX][ANCHOR_SIZE];

  guint x_scale;
  guint y_scale;
  guint h_scale;
  guint w_scale;

  guint i_width;
  guint i_height;

  gfloat min_score_thresh;

  gfloat iou_thresh;
} BlazeFaceInfo;

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

  GstElement *pipeline; /**< gst pipeline for data stream */

  GMainLoop *loop; /**< main event loop */
  GstBus *bus; /**< gst bus for data pipeline */

  guint crop_received;
} AppData;

/** @brief Represents a detect object */
typedef struct
{
  int valid;
  int class_id;
  int x;
  int y;
  int width;
  int height;
  gfloat prob;
} detectedObject;

/**
 * @brief Calculate the intersected surface
 */
static gfloat
iou (detectedObject * a, detectedObject * b)
{
  int x1 = MAX (a->x, b->x);
  int y1 = MAX (a->y, b->y);
  int x2 = MIN (a->x + a->width, b->x + b->width);
  int y2 = MIN (a->y + a->height, b->y + b->height);
  int w = MAX (0, (x2 - x1 + 1));
  int h = MAX (0, (y2 - y1 + 1));
  float inter = w * h;
  float areaA = a->width * a->height;
  float areaB = b->width * b->height;
  float o = inter / (areaA + areaB - inter);
  return (o >= 0) ? o : 0;
}

/**
 * @brief Compare Function for g_array_sort with detectedObject.
 */
static gint
compare_detection (gconstpointer _a, gconstpointer _b)
{
  const detectedObject *a = _a;
  const detectedObject *b = _b;

  /* Larger comes first */
  return (a->prob > b->prob) ? -1 : ((a->prob == b->prob) ? 0 : 1);
}

/**
 * @brief Apply NMS to the given results (objects[MOBILENET_SSD_DETECTION_MAX])
 * @param[in/out] results The results to be filtered with nms
 */
static void
nms (GArray * results, gfloat threshold)
{
  guint boxes_size;
  guint i, j;

  g_array_sort (results, compare_detection);
  boxes_size = results->len;

  for (i = 0; i < boxes_size; i++) {
    detectedObject *a = &g_array_index (results, detectedObject, i);
    if (a->valid == TRUE) {
      for (j = i + 1; j < boxes_size; j++) {
        detectedObject *b = &g_array_index (results, detectedObject, j);
        if (b->valid == TRUE) {
          if (iou (a, b) > threshold) {
            b->valid = FALSE;
          }
        }
      }
    }
  }

  i = 0;
  do {
    detectedObject *a = &g_array_index (results, detectedObject, i);
    if (a->valid == FALSE)
      g_array_remove_index (results, i);
    else
      i++;
  } while (i < results->len);
}

static gboolean
convert_to_detection (BlazeFaceInfo *info, float* detection_boxes, float* detection_scores, GArray *result)
{
  for (int i = 0; i < info->num_boxes; i++) {

  }
  return TRUE;
}

gboolean
get_detected_object_i (int i, float* raw_boxes, float* raw_scores, BlazeFaceInfo *info, detectedObject *object)
{
  int box_offset = i * BLAZEFACE_NUM_COORD;

  float x_center = raw_boxes[box_offset];
  float y_center = raw_boxes[box_offset+1];
  float w = raw_boxes[box_offset+2];
  float h = raw_boxes[box_offset+3];
  float score = raw_scores[i];
  float ymin, xmin, ymax, xmax;
  int x, y, width, height;

  // decode boxes
  x_center = x_center / info->x_scale * info->anchors[i][ANCHOR_WIDTH_IDX] + info->anchors[i][ANCHOR_X_CENTER_IDX];
  y_center = y_center / info->y_scale * info->anchors[i][ANCHOR_HEIGHT_IDX] + info->anchors[i][ANCHOR_Y_CENTER_IDX];

  h = h / info->h_scale * info->anchors[i][ANCHOR_HEIGHT_IDX];
  w = w / info->w_scale * info->anchors[i][ANCHOR_WIDTH_IDX];

  ymin = y_center - h / 2.0f;
  xmin = x_center - w / 2.0f;
  ymax = y_center + h / 2.0f;
  xmax = x_center + w / 2.0f;

  // decode score
  score = score < -100.0 ? -100 : score;
  score = score > 100.0 ? 100 : score;
  score = 1.0f / (1.0f + expf(-score));

  x = xmin * info->i_width;
  y = ymin * info->i_height;
  width = (xmax - xmin) * info->i_width;
  height = (ymax - ymin) * info->i_height;
  object->x = MAX(0, x);
  object->y = MAX(0, y);
  object->width = MIN(width, info->i_width - object->x);
  object->height = MIN(height, info->i_height - object->y);
  object->prob = score;
  object->valid = score >= 0.5f;

  return TRUE;
}

static void
crop_new_data_cb (GstElement * element, GstBuffer * buffer, AppData *app)
{
  gsize raw_boxes_size, raw_scores_size;
  float *raw_boxes, *raw_scores;
  BlazeFaceInfo *info = &app->detect_model;

  app->crop_received++;
  if (TRUE || app->crop_received % 90 == 0) {
    _print_log ("CROP::receiving new data [%d]", app->crop_received);
  }

  {
    GstMemory *mem;
    GstMapInfo info;
    guint i;
    guint num_mems;
    GstTensorMetaInfo meta;

    num_mems = gst_buffer_n_memory (buffer);

    _print_log ("num_mems %u", num_mems);
    for (int i = 0; i < num_mems; i++) {
      mem = gst_buffer_peek_memory (buffer, 0);
      gst_tensor_meta_info_parse_memory (&meta, mem);
      _print_log("tensor meta info type: %d", meta.type);
      _print_log("tensor meta info format: %d", meta.format);
      _print_log("tensor meta info media_type: %d", meta.media_type);
      _print_log("tensor meta info dimension: %d:%d:%d", meta.dimension[0], meta.dimension[1], meta.dimension[2]);
      if (gst_memory_map (mem, &info, GST_MAP_READ)) {
        /* check data (info.data, info.size) */
        _print_log ("received %zd", info.size);
        gst_memory_unmap (mem, &info);
      }
    }
  }

  {
    GstPad *sink_pad;
    GstCaps *caps;

    sink_pad = gst_element_get_static_pad (element, "sink");

    if (sink_pad) {
      caps = gst_pad_get_current_caps (sink_pad);

      if (caps) {
        _parse_caps (caps);
        gst_caps_unref (caps);
      }
    }
  }
}

gboolean
build_pipeline (AppData *app)
{
  GstElement *tee_source, *compositor;
  GstPad *cropinfo_srcpad, *cropped_video_srcpad, *landmark_overray_srcpad;

  app->pipeline = gst_pipeline_new ("facemesh-pipeline");
  if (!app->pipeline) {
    g_printerr ("pipeline could not be created.\n");
    return FALSE;
  }

  /* Souece video */
  {
    GstElement *video_source, *convert_source, *filter, *crop_source;
    GstCaps *convert_caps;

    video_source = gst_element_factory_make ("v4l2src", "video_source");
    convert_source = gst_element_factory_make ("videoconvert", "convert_source");
    filter = gst_element_factory_make ("capsfilter", "filter1");
    crop_source = gst_element_factory_make ("videocrop", "crop_source");
    tee_source = gst_element_factory_make ("tee", "tee_source");

    if (!video_source || !convert_source || !filter || !crop_source || !tee_source) {
      g_printerr ("[SOUECE] Not all elements could be created.\n");
      return FALSE;
    }

    convert_caps = gst_caps_from_string ("video/x-raw,format=RGB,width=1280,height=720,framerate=30/1");
    g_object_set (G_OBJECT (filter), "caps", convert_caps, NULL);
    gst_caps_unref (convert_caps);

    g_object_set (crop_source, "left", 280, "right", 280, NULL);

    gst_bin_add_many (GST_BIN (app->pipeline), video_source, convert_source, filter, crop_source, tee_source, NULL);
    if (!gst_element_link_many (video_source, convert_source, filter, crop_source, tee_source, NULL)) {
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

    queue = gst_element_factory_make ("queue", "queue_detect");
    scale = gst_element_factory_make ("videoscale", "scale_detect");
    filter = gst_element_factory_make ("capsfilter", "filter_detect");
    tconv = gst_element_factory_make ("tensor_converter", "tconv_detect");
    ttransform = gst_element_factory_make ("tensor_transform", "ttransform_detect");
    tfilter_detect = gst_element_factory_make ("tensor_filter", "tfilter_detect");
    tfilter_cropinfo = gst_element_factory_make ("tensor_filter", "filter_cropinfo");

    if (!queue || !scale || !filter || !tconv || !ttransform || !tfilter_detect || !tfilter_cropinfo) {
      g_printerr ("[DETECT] Not all elements could be created.\n");
      return FALSE;
    }

    scale_caps = gst_caps_from_string ("video/x-raw,format=RGB,width=128,height=128");
    g_object_set (G_OBJECT (filter), "caps", scale_caps, NULL);
    gst_caps_unref (scale_caps);

    g_object_set (ttransform, "mode", 2 /* GTT_ARITHMETIC */, "option", "typecast:float32,add:-127.5,div:127.5", NULL);
    g_object_set (tfilter_detect, "framework", "tensorflow-lite", "model", app->detect_model.model_path, NULL);
    g_object_set (tfilter_cropinfo, "framework", "custom-easy", "model", "detection_to_cropinfo", NULL);

    gst_bin_add_many (GST_BIN (app->pipeline), 
        queue, scale, filter, tconv, ttransform, tfilter_detect, tfilter_cropinfo, NULL);

    if (!gst_element_link_many (queue, scale, filter, tconv, ttransform, tfilter_detect, tfilter_cropinfo, NULL)) {
      g_printerr ("[DETECT] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }

    tee_pad = gst_element_request_pad_simple (tee_source, "src_%u");
    g_print ("[DETECT] Obtained request pad %s\n", gst_pad_get_name (tee_pad));
    queue_pad = gst_element_get_static_pad (queue, "sink");

    if (gst_pad_link (tee_pad, queue_pad) != GST_PAD_LINK_OK) {
      g_printerr ("[DETECT] Tee could not be linked\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    gst_object_unref (queue_pad);

    cropinfo_srcpad = gst_element_get_static_pad (tfilter_cropinfo, "src");
  }

  /* Crop video */
  {
    GstElement *queue, *tconv, *tcrop;
    GstPad *cropinfo_sinkpad, *tee_pad, *queue_pad;

    queue = gst_element_factory_make ("queue", "queue_crop");
    tconv = gst_element_factory_make ("tensor_converter", "tconv_crop");
    tcrop = gst_element_factory_make ("tensor_crop", "tcrop");

    if (!queue || !tconv || !tcrop) {
      g_printerr ("[CROP] Not all elements could be created.\n");
      return FALSE;
    }

    gst_bin_add_many (GST_BIN (app->pipeline), queue, tconv, tcrop, NULL);

    cropinfo_sinkpad = gst_element_get_static_pad (tcrop, "info");
    if (!gst_element_link_many (queue, tconv, NULL)
        || !gst_element_link_pads (tconv, "src", tcrop, "raw")
        || gst_pad_link (cropinfo_srcpad, cropinfo_sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("[CROP] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }

    tee_pad = gst_element_request_pad_simple (tee_source, "src_%u");
    g_print ("[CROP] Obtained request pad %s\n", gst_pad_get_name (tee_pad));
    queue_pad = gst_element_get_static_pad (queue, "sink");

    if (gst_pad_link (tee_pad, queue_pad) != GST_PAD_LINK_OK) {
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
    g_object_set (tdec_landmark, "mode", "face_mesh", "option1", "mediapipe-face-mesh", "option2", "720:720", "option3", "192:192", NULL);

    gst_bin_add_many (GST_BIN (app->pipeline),
        queue, tdec_flexible, tconv, ttransform, tfilter_landmark, tdec_landmark, NULL);

    cropped_video_sinkpad = gst_element_get_static_pad (queue, "sink");
    cropped_video_caps = gst_caps_from_string ("video/x-raw,format=RGB,width=192,height=192,framerate=30/1");
    if (gst_pad_link (cropped_video_srcpad, cropped_video_sinkpad) != GST_PAD_LINK_OK
        || !gst_element_link (queue, tdec_flexible)
        || !gst_element_link_filtered (tdec_flexible, tconv, cropped_video_caps)
        || !gst_element_link_many (tconv, ttransform, tfilter_landmark, tdec_landmark, NULL))
    {
      g_printerr ("[LANDMARK] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    g_object_unref (cropped_video_caps);

    landmark_overray_srcpad = gst_element_get_static_pad (tdec_landmark, "src");
  }

  /* Result video */
  {
    GstElement *queue, *compositor, *convert, *video_sink;
    GstPad *tee_pad, *queue_pad, *compositor_overray_pad, *compositor_video_pad;

    queue = gst_element_factory_make ("queue", "queue_result");
    compositor = gst_element_factory_make ("compositor", "compositor");
    convert = gst_element_factory_make ("videoconvert", "convert_result");
    video_sink = gst_element_factory_make ("autovideosink", "video_sink");

    if (!queue || !compositor || !convert || !video_sink) {
      g_printerr ("[RESULT] Not all elements could be created.\n");
      return FALSE;
    }

    gst_bin_add_many (GST_BIN (app->pipeline), queue, compositor, convert, video_sink, NULL);
    
    if (!gst_element_link_many (compositor, convert, video_sink, NULL)) {
      g_printerr ("[RESULT] Elements could not be linked.\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }

    compositor_overray_pad = gst_element_request_pad_simple (compositor, "sink_%u");
    g_print ("[RESULT] Obtained request pad %s\n", gst_pad_get_name (compositor_overray_pad));
    g_object_set (compositor_overray_pad, "zorder", 2, NULL);

    compositor_video_pad = gst_element_request_pad_simple (compositor, "sink_%u");
    g_print ("[RESULT] Obtained request pad %s\n", gst_pad_get_name (compositor_video_pad));
    g_object_set (compositor_video_pad, "zorder", 1, NULL);
    queue_pad = gst_element_get_static_pad (queue, "src");

    if (gst_pad_link (landmark_overray_srcpad, compositor_overray_pad) != GST_PAD_LINK_OK
        || gst_pad_link (queue_pad, compositor_video_pad) != GST_PAD_LINK_OK) {
      g_printerr ("[RESULT] Compositor could not be linked\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    gst_object_unref (queue_pad);

    tee_pad = gst_element_request_pad_simple (tee_source, "src_%u");
    g_print ("[RESULT] Obtained request pad %s\n", gst_pad_get_name (tee_pad));
    queue_pad = gst_element_get_static_pad (queue, "sink");

    if (gst_pad_link (tee_pad, queue_pad) != GST_PAD_LINK_OK) {
      g_printerr ("[RESULT] Tee could not be linked\n");
      gst_object_unref (app->pipeline);
      return FALSE;
    }
    gst_object_unref (queue_pad);
  }

  gst_object_unref (cropinfo_srcpad);

  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN (app->pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");
  return TRUE;
}

/**
 * @brief Load box-prior data from a file
 * @param[in/out] info The BlazeFace model info
 * @return TRUE if loaded and configured. FALSE if failed to do so.
 */
static int
blazeface_load_anchors (BlazeFaceInfo *info)
{
  gboolean failed = FALSE;
  GError *err = NULL;
  gchar **priors;
  gchar *line = NULL;
  gchar *contents = NULL;
  guint row;
  gint prev_reg = -1;

  /* Read file contents */
  if (!g_file_get_contents (info->anchors_path, &contents, NULL, &err)) {
    GST_ERROR ("BlazeFace's box prior file %s cannot be read: %s",
        info->anchors_path, err->message);
    g_clear_error (&err);
    return FALSE;
  }

  priors = g_strsplit (contents, "\n", -1);
  /* If given prior file is inappropriate, report back to tensor-decoder */
  if (g_strv_length (priors) < ANCHOR_SIZE) {
    g_critical ("The given prior file, %s, should have at least %d lines.\n",
        info->anchors_path, ANCHOR_SIZE);
    failed = TRUE;
    goto error;
  }

  for (row = 0; row < ANCHOR_SIZE; row++) {
    gint column = 0, registered = 0;

    line = priors[row];
    if (line) {
      gchar **list = g_strsplit_set (line, " \t,", -1);
      gchar *word;

      while ((word = list[column]) != NULL) {
        column++;

        if (word && *word) {
          if (registered > DETECTION_MAX) {
            GST_WARNING
                ("BlazeFace's box prior data file has too many priors. %d >= %d",
                registered, DETECTION_MAX);
            break;
          }
          info->anchors[registered][row] =
              (gfloat) g_ascii_strtod (word, NULL);
          registered++;
        }
      }

      g_strfreev (list);
    }

    if (prev_reg != -1 && prev_reg != registered) {
      GST_ERROR
          ("BlazeFace's box prior data file is not consistent.");
      failed = TRUE;
      break;
    }
    prev_reg = registered;
  }

error:
  g_strfreev (priors);
  g_free (contents);
  return !failed;
}
static void
margin_object(detectedObject *orig, detectedObject *margined, float margin_rate)
{
  int height = orig->height;
  int width = orig->width;
  int orig_size = MAX(height, width);
  int margin = orig_size * margin_rate;
  int size = MIN(orig_size + margin * 2, 720);
  int x = MIN(MAX(orig->x - margin, 0), 720 - size);
  int y = MIN(MAX(orig->y - margin, 0), 720 - size);
  margined->x = x;
  margined->y = y;
  margined->width = size;
  margined->height = size;
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
    _print_log ("no detected object");
    info_data[0] = 0U;
    info_data[1] = 0U;
    info_data[2] = info->i_width;
    info_data[3] = info->i_height;
  } else {
    detectedObject *object = &g_array_index (results, detectedObject, 0);
    detectedObject margined;

    margin_object (object, &margined, 0.25);
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

  size_t size = _get_video_xraw_RGB_bufsize (192, 192);

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
  for (h = 0; h < 192; h++) {
    int h_inp = (int)((float)dim[2] / 192 * h);
    uint8_t *row_inp = inp + dim[0] * dim[1] * h_inp;
    uint8_t *row_ptr = ptr;
    for (w = 0; w < 192; w++) {
      int w_inp = (int)((float)dim[1] / 192 * w);
      uint8_t *pix_inp = row_inp + dim[0] * w_inp;
      memcpy (row_ptr, pix_inp, 3);
      row_ptr += 3;
    }
    ptr += ((3 * 192 - 1) / 4 + 1) * 4;
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
init_blazeface (BlazeFaceInfo *info, const gchar *path)
{
  const gchar detect_model[] = "face_detection_short_range.tflite";
  const gchar detect_box_prior[] = "box_prior.txt";
  const guint num_boxs = BLAZEFACE_SHORT_RANGE_NUM_BOXS;

  info->model_path = g_strdup_printf ("%s/%s", path, detect_model);
  info->anchors_path = g_strdup_printf ("%s/%s", path, detect_box_prior);
  info->num_boxes = num_boxs;
  info->x_scale = 128;
  info->y_scale = 128;
  info->h_scale = 128;
  info->w_scale = 128;
  info->i_width = 720;
  info->i_height = 720;
  info->min_score_thresh = 0.5f;
  info->iou_thresh = 0.3f;

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
init_landmark_model (LandmarkModelInfo *info, const gchar *path)
{
  const gchar landmark_model[] = "face_landmark.tflite";

  info->model_path = g_strdup_printf ("%s/%s", path, landmark_model);
  info->tensor_width = 192;
  info->tensor_height = 192;
  info->i_width = 720;
  info->i_height = 720;

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

  init_blazeface (&app->detect_model, resource_path);
  init_landmark_model (&app->landmark_model, resource_path);
  app->crop_received = 0;

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
  int ret = NNS_custom_easy_register ("detection_to_cropinfo", cef_func_detection_to_cropinfo, app, &info_in, &info_out);

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
  GstState old_state, new_state, pending_state;
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
    case GST_MESSAGE_STATE_CHANGED:
      gst_message_parse_state_changed (msg, &old_state, &new_state, &pending_state);
      g_print ("%s state changed from %s to %s:\n", GST_OBJECT_NAME (msg->src), gst_element_state_get_name (old_state), gst_element_state_get_name (new_state));
      break;
    case GST_MESSAGE_QOS:
      break;
    default:
      g_printerr ("msg from %s:  %s\n", GST_OBJECT_NAME (msg->src), gst_message_type_get_name (GST_MESSAGE_TYPE (msg)));
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
