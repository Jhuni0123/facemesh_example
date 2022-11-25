#include <gst/gst.h>
#include <nnstreamer/nnstreamer_plugin_api_decoder.h>
#include <nnstreamer/nnstreamer_plugin_api_util.h>
#include <nnstreamer/nnstreamer_plugin_api.h>
#include <nnstreamer/tensor_decoder_custom.h>
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

/**
 * @brief Data structure for app.
 */
typedef struct
{
  BlazeFaceInfo detect_model;

  GstElement *pipeline; /**< gst pipeline for data stream */

  GMainLoop *loop; /**< main event loop */
  GstBus *bus; /**< gst bus for data pipeline */

  guint detect_received;
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
  GstElement *video_source, *video_convert1, *filter1, *video_crop;
  GstElement *video_convert2, *video_sink;
  GstElement *video_scale1, *filter2, *tensor_converter, *tensor_transform, *tensor_filter, *custom_filter_detection;
  GstElement *tensor_converter_crop, *tensor_crop;
  GstElement *tee, *queue_crop, *queue_detect, *queue_result;
  GstElement *crop_sink;
  GstElement *tensor_decoder_video;
  GstElement *video_scale_crop, *video_convert_crop, *video_sink_crop;

  app->pipeline = gst_pipeline_new ("face-crop-pipeline");

  /* Source video */
  video_source = gst_element_factory_make ("v4l2src", "video_source");
  video_convert1 = gst_element_factory_make ("videoconvert", "video_convert1");
  filter1 = gst_element_factory_make ("capsfilter", "filter1");
  video_crop = gst_element_factory_make ("videocrop", "video_crop");

  tee = gst_element_factory_make ("tee", "tee");

  /* Face Detect */
  queue_detect = gst_element_factory_make ("queue", "queue_detect");
  video_scale1 = gst_element_factory_make ("videoscale", "video_scale1");
  filter2 = gst_element_factory_make ("capsfilter", "filter2");
  tensor_converter = gst_element_factory_make ("tensor_converter", "tensor_converter");
  tensor_transform = gst_element_factory_make ("tensor_transform", "tensor_transform");
  tensor_filter = gst_element_factory_make ("tensor_filter", "tensor_filter");
  custom_filter_detection = gst_element_factory_make ("tensor_filter", "custom_filter_detection");

  /* Crop video */
  queue_crop = gst_element_factory_make ("queue", "queue_crop");
  tensor_converter_crop = gst_element_factory_make ("tensor_converter", "tensor_converter_crop");
  tensor_crop = gst_element_factory_make ("tensor_crop", "tensor_crop");
  crop_sink = gst_element_factory_make ("tensor_sink", "crop_sink");
  //tensor_decoder_video = gst_element_factory_make ("tensor_decoder", "tensor_decoder_video");
  //video_scale_crop = gst_element_factory_make ("videoscale", "video_scale_crop");
  //video_convert_crop = gst_element_factory_make ("videoconvert", "video_convert_crop");
  //video_sink_crop = gst_element_factory_make ("autovideosink", "video_sink_crop");


  /* Result */
  queue_result = gst_element_factory_make ("queue", "queue_result");
  video_convert2 = gst_element_factory_make ("videoconvert", "video_convert2");
  video_sink = gst_element_factory_make ("autovideosink", "video_sink");

  if (!app->pipeline || !video_source || !video_convert1 || !filter1 || !video_crop || !tee
      || !queue_crop || !tensor_converter_crop || !tensor_crop
      || !queue_result || !video_convert2 || !video_sink
      //|| !tensor_decoder_video || !video_scale_crop || !video_convert_crop || !video_sink_crop
      || !crop_sink
      || !queue_detect || !video_scale1 || !filter2 || !tensor_converter || !tensor_transform || !tensor_filter || !custom_filter_detection) {
    g_printerr ("Not all elements could be created.\n");
    return FALSE;
  }

  /* Set properties */
  GstCaps *convert_caps, *scale_caps;

  convert_caps = gst_caps_from_string ("video/x-raw,format=RGB,width=1280,height=720,framerate=30/1");
  g_object_set (G_OBJECT (filter1), "caps", convert_caps, NULL);
  gst_caps_unref (convert_caps);

  g_object_set (video_crop, "left", 280, "right", 280, NULL);

  scale_caps = gst_caps_from_string ("video/x-raw,format=RGB,width=128,height=128");
  g_object_set (G_OBJECT (filter2), "caps", scale_caps, NULL);
  gst_caps_unref (scale_caps);

  g_object_set (tensor_transform, "mode", 2 /* GTT_ARITHMETIC */, "option", "typecast:float32,add:-127.5,div:127.5", NULL);

  g_object_set (tensor_filter, "framework", "tensorflow-lite", "model", app->detect_model.model_path, NULL);
  g_object_set (custom_filter_detection, "framework", "custom-easy", "model", "detection_to_cropinfo", NULL);

  g_signal_connect (crop_sink, "new-data", (GCallback) crop_new_data_cb, app);

  //g_object_set (tensor_decoder_video, "mode", "direct_video", NULL);


  /* Link all "Always" pads */
  gst_bin_add_many (GST_BIN (app->pipeline), video_source, video_convert1, filter1, video_crop, tee,
      queue_crop, tensor_converter_crop, tensor_crop,
      //tensor_decoder_video, video_scale_crop, video_convert_crop, video_sink_crop,
      crop_sink,
      queue_result, video_convert2, video_sink,
      queue_detect, video_scale1, filter2, tensor_converter, tensor_transform, tensor_filter, custom_filter_detection, NULL);

  if (!gst_element_link_many (video_source, video_convert1, filter1, video_crop, tee, NULL)
      || !gst_element_link_many (queue_detect, video_scale1, filter2, tensor_converter, tensor_transform, tensor_filter, custom_filter_detection, NULL)
      || !gst_element_link_many (queue_crop, tensor_converter_crop, NULL)
      || !gst_element_link_pads (tensor_converter_crop, "src", tensor_crop, "raw")
      || !gst_element_link_pads (custom_filter_detection, "src", tensor_crop, "info")
      //|| !gst_element_link_many (tensor_crop, tensor_decoder_video, video_scale_crop, video_convert_crop, video_sink_crop, NULL)
      || !gst_element_link_many (tensor_crop, crop_sink, NULL)
      || !gst_element_link_many (queue_result, video_convert2, video_sink, NULL)
  ) {
    g_printerr ("Elements could not be linked.\n");
    gst_object_unref (app->pipeline);
    return FALSE;
  }

  /* Link tee's "Request" pads */
  GstPad *tee_detect_pad, *tee_crop_pad, *tee_result_pad;
  GstPad *queue_detect_pad, *queue_crop_pad, *queue_result_pad;

  tee_crop_pad = gst_element_request_pad_simple (tee, "src_%u");
  g_print ("Obtained request pad %s for crop branch.\n", gst_pad_get_name (tee_crop_pad));
  queue_crop_pad = gst_element_get_static_pad (queue_crop, "sink");

  tee_detect_pad = gst_element_request_pad_simple (tee, "src_%u");
  g_print ("Obtained request pad %s for detect branch.\n", gst_pad_get_name (tee_detect_pad));
  queue_detect_pad = gst_element_get_static_pad (queue_detect, "sink");

  tee_result_pad = gst_element_request_pad_simple (tee, "src_%u");
  g_print ("Obtained request pad %s for result branch.\n", gst_pad_get_name (tee_result_pad));
  queue_result_pad = gst_element_get_static_pad (queue_result, "sink");

  if (gst_pad_link (tee_crop_pad, queue_crop_pad) != GST_PAD_LINK_OK
      || gst_pad_link (tee_detect_pad, queue_detect_pad) != GST_PAD_LINK_OK
      || gst_pad_link (tee_result_pad, queue_result_pad) != GST_PAD_LINK_OK
  ) {
    g_printerr ("Tee could not be linked\n");
    gst_object_unref (app->pipeline);
    return FALSE;
  }
  gst_object_unref (queue_crop_pad);
  gst_object_unref (queue_detect_pad);
  gst_object_unref (queue_result_pad);

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
    _print_log ("detected: %d %d %d %d = %d", margined.x, margined.y, margined.height, margined.width, margined.height * margined.width * 3);

    info_data[0] = margined.x;
    info_data[1] = margined.y;
    info_data[2] = margined.width;
    info_data[3] = margined.height;
  }
  return 0;
}

int flexible_tensor_to_video (const GstTensorMemory *input, const GstTensorsConfig *config, void *data, GstBuffer *out_buf) {
  AppData *app = data;

  return 0;
}

gboolean 
init_app (AppData *app)
{
  const gchar resource_path[] = "./res";

  init_blazeface (&app->detect_model, resource_path);
  app->detect_received = 0;
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
