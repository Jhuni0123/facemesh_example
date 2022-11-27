#include <gst/gst.h>
#include <math.h>

#define BLAZEFACE_SHORT_RANGE_NUM_BOXS  (896)
#define BLAZEFACE_NUM_COORD             (16)

/**
 * @brief Sigmoid function
 */
#define sigmoid(x) \
    (1.f / (1.f + expf (- ((float)x))))

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
  score = sigmoid(score);

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

