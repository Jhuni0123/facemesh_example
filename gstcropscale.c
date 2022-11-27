/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2022 Parallels <<user@hostname.org>>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * SECTION:element-cropscale
 *
 * FIXME:Describe cropscale here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! cropscale ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gst/gst.h>
#include <gst/video/video-format.h>
#include <gst/video/gstvideoaggregator.h>
#include <nnstreamer/tensor_typedef.h>
#include <nnstreamer/nnstreamer_plugin_api_util.h>
#include <nnstreamer/nnstreamer_plugin_api.h>
#include <nnstreamer/nnstreamer_plugin_api_filter.h>
#include <nnstreamer/nnstreamer_util.h>

#include "gstcropscale.h"

/**
 * @brief Internal data structure to describe tensor region.
 */
typedef struct
{
  guint x;
  guint y;
  guint w;
  guint h;
} tensor_crop_info_s;

GST_DEBUG_CATEGORY_STATIC (gst_crop_scale_debug);
#define GST_CAT_DEFAULT gst_crop_scale_debug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate raw_factory = GST_STATIC_PAD_TEMPLATE ("raw",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE ("RGBA"))
    );

static GstStaticPadTemplate info_factory = GST_STATIC_PAD_TEMPLATE ("info",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT)
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE ("RGBA"))
    );

#define gst_crop_scale_parent_class parent_class
G_DEFINE_TYPE (GstCropScale, gst_crop_scale, GST_TYPE_ELEMENT);

GST_ELEMENT_REGISTER_DEFINE (crop_scale, "crop_scale", GST_RANK_NONE,
    GST_TYPE_CROP_SCALE);

static void gst_crop_scale_finalize (GObject *object);
static void gst_crop_scale_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_crop_scale_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static GstStateChangeReturn gst_crop_scale_change_state (GstElement *element,
    GstStateChange transition);
static gboolean gst_crop_scale_src_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_crop_scale_sink_event (GstCollectPads *pads,
    GstCollectData *data, GstEvent *event, gpointer user_data);
static GstFlowReturn gst_crop_scale_collected (GstCollectPads *pads,
    gpointer user_data);

/* GObject vmethod implementations */

/* initialize the cropscale's class */
static void
gst_crop_scale_class_init (GstCropScaleClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_crop_scale_set_property;
  gobject_class->get_property = gst_crop_scale_get_property;
  gobject_class->finalize = gst_crop_scale_finalize;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gstelement_class->change_state =
    GST_DEBUG_FUNCPTR (gst_crop_scale_change_state);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&raw_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&info_factory));

  gst_element_class_set_details_simple (gstelement_class,
      "CropScale",
      "FIXME:Generic",
      "FIXME:Generic Template Element", "Parallels <<user@hostname.org>>");

}

/**
 * @brief Clear and reset old pad data.
 */
static void
gst_crop_scale_pad_reset (GstCropScalePadData * cpad)
{
}

/**
 * @brief Clear and reset old data in crop_scale
 */
static void
gst_crop_scale_reset (GstCropScale * self)
{
  GstCropScalePadData *cpad;
  GSList *walk;

  if (self->collect) {
    walk = self->collect->data;

    while (walk) {
      cpad = (GstCropScalePadData *) walk->data;

      gst_crop_scale_pad_reset (cpad);
      walk = g_slist_next (walk);
    }
  }

  self->send_stream_start = TRUE;
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad callback functions
 * initialize instance structure
 */
static void
gst_crop_scale_init (GstCropScale * self)
{
  self->sinkpad_raw = gst_pad_new_from_static_template (&raw_factory, "raw");
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad_raw);

  self->sinkpad_info = gst_pad_new_from_static_template (&info_factory, "info");
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad_info);

  self->collect = gst_collect_pads_new ();
  gst_collect_pads_set_function (self->collect,
      GST_DEBUG_FUNCPTR (gst_crop_scale_collected), self);
  gst_collect_pads_set_event_function (self->collect,
      GST_DEBUG_FUNCPTR (gst_crop_scale_sink_event), self);

  gst_collect_pads_add_pad (self->collect, self->sinkpad_raw,
      sizeof (GstCropScalePadData), NULL, TRUE);
  gst_collect_pads_add_pad (self->collect, self->sinkpad_info,
      sizeof (GstCropScalePadData), NULL, TRUE);

  self->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  gst_pad_set_event_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_crop_scale_src_event));
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  self->silent = FALSE;
  self->send_stream_start = TRUE;
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_crop_scale_finalize (GObject * object)
{
  GstCropScale *self;

  self = GST_CROP_SCALE (object);

  gst_crop_scale_reset (self);

  if (self->collect) {
    gst_object_unref (self->collect);
    self->collect = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static void
gst_crop_scale_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCropScale *filter = GST_CROP_SCALE (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_crop_scale_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstCropScale *filter = GST_CROP_SCALE (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


/* GstElement vmethod implementations */

/**
 * @brief Handle state transition.
 */
static GstStateChangeReturn
gst_crop_scale_change_state (GstElement * element, GstStateChange transition)
{
  GstCropScale *self;
  GstStateChangeReturn ret;

  self = GST_CROP_SCALE (element);

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_collect_pads_start (self->collect);
      break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_collect_pads_stop (self->collect);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_crop_scale_reset (self);
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief Handle event on src pad.
 */
static gboolean
gst_crop_scale_src_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  g_return_val_if_fail (event != NULL, FALSE);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEEK:
      /* disable seeking */
      gst_event_unref (event);
      return FALSE;
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/* this function handles sink events */
static gboolean
gst_crop_scale_sink_event (GstCollectPads *pads, GstCollectData *data,
    GstEvent * event, gpointer user_data)
{
  GstCropScalePadData *cpad;

  g_return_val_if_fail (event != NULL, FALSE);

  cpad = (GstCropScalePadData *) data;

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      GstStructure *structure;
      gboolean ret;

      gst_event_parse_caps (event, &caps);

      // FIXME: which pad the event came from

      // raw
      ret = gst_video_info_from_caps (&cpad->info, caps);
      if (ret) {
        cpad->is_raw = TRUE;
        gst_event_unref (event);
        return ret;
      }

      // info
      structure = gst_caps_get_structure (caps, 0);

      gst_tensors_config_from_structure (&cpad->config, structure);

      gst_event_unref (event);
      return TRUE;
      return gst_tensors_config_validate (&cpad->config);
      
    }
    default:
      break;
  }

  return gst_collect_pads_event_default (pads, data, event, FALSE);
}

/**
 * @brief Set pad caps if not negotiated.
 */
static GstFlowReturn
gst_crop_scale_negotiate (GstCropScale * self)
{
  if (!gst_pad_has_current_caps (self->sinkpad_raw)) {
    GST_ERROR_OBJECT (self,
        "The raw pad of crop_scale '%s' does not have pad caps.",
        GST_ELEMENT_NAME (self));
    return GST_FLOW_NOT_NEGOTIATED;
  }

  if (!gst_pad_has_current_caps (self->sinkpad_info)) {
    GST_ERROR_OBJECT (self,
        "The info pad of crop_scale '%s' does not have pad caps.",
        GST_ELEMENT_NAME (self));
    return GST_FLOW_NOT_NEGOTIATED;
  }

  if (!gst_pad_has_current_caps (self->srcpad)) {
    GstCaps *caps;
    GstSegment segment;
    GstCropScalePadData *cpad;
    GSList *walk;
    gint fps_d, fps_n, width, height;

    UNUSED (cpad);

    if (self->send_stream_start) {
      gchar *sid;

      sid = g_strdup_printf ("%s-%08x",
          GST_ELEMENT_NAME (self), g_random_int ());
      gst_pad_push_event (self->srcpad, gst_event_new_stream_start (sid));
      g_free (sid);

      self->send_stream_start = FALSE;
    }

    /**
     * Get video info from collect-pads and set framerate.
     * Output is always video/x-raw.
     */

    fps_n = -1;
    fps_d = -1;

    walk = self->collect->data;
    while (walk) {
      cpad = (GstCropScalePadData *) walk->data;
      if (cpad->is_raw) {
        GST_DEBUG ("fps: %d/%d", cpad->info.fps_n, cpad->info.fps_d);
        if (fps_n < 0 ||
            gst_util_fraction_compare (cpad->info.fps_n, cpad->info.fps_d,
              fps_n, fps_d) < 0) {
          fps_n = cpad->info.fps_n;
          fps_d = cpad->info.fps_d;
          width = cpad->info.width;
          height = cpad->info.height;
        }
      } else {
        GST_DEBUG ("rate: %d/%d", cpad->config.rate_n, cpad->config.rate_d);
      }

      walk = g_slist_next (walk);
    }
    caps = gst_caps_new_simple ("video/x-raw",
       "format", G_TYPE_STRING, "RGBA",
       "framerate", GST_TYPE_FRACTION, fps_n, fps_d,
       "width", G_TYPE_INT, width,
       "height", G_TYPE_INT, height,
       NULL);
    gst_pad_set_caps (self->srcpad, caps);
    gst_caps_unref (caps);

    gst_segment_init (&segment, GST_FORMAT_TIME);
    gst_pad_push_event (self->srcpad, gst_event_new_segment (&segment));
  }

  return GST_FLOW_OK;
}

/**
 * @brief Internal function to parse buffer and fill crop info.
 */
static gboolean
gst_crop_scale_get_crop_info (GstCropScale * self, GstBuffer * info,
    tensor_crop_info_s * cinfo)
{
  GstMemory *mem;
  GstMapInfo map;
  GstTensorMetaInfo meta;
  gsize hsize, dsize, esize;
  guint i;
  guint *pos;
  gboolean ret = FALSE;

  i = gst_buffer_n_memory (info);
  g_assert (i > 0);
  if (i > 1) {
    GST_WARNING_OBJECT (self,
        "Info buffer has %u memories, parse first one.", i);
  }

  mem = gst_buffer_peek_memory (info, 0);
  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    GST_ERROR_OBJECT (self, "Failed to map the info buffer.");
    return FALSE;
  }

  /* parse crop-info from flex tensor */
  if (!gst_tensor_meta_info_parse_header (&meta, map.data)) {
    GST_ERROR_OBJECT (self, "Failed to get the meta from info buffer.");
    goto done;
  }

  hsize = gst_tensor_meta_info_get_header_size (&meta);
  dsize = gst_tensor_meta_info_get_data_size (&meta);
  esize = gst_tensor_get_element_size (meta.type);

  if (hsize + dsize != map.size) {
    GST_ERROR_OBJECT (self,
        "Invalid meta info, info buffer size is incorrect (received %zd, expected %zd).",
        map.size, hsize + dsize);
    goto done;
  }

  /**
   * @todo Add various mode to crop tensor.
   * Now tensor-crop handles NHWC data format only.
   */
  g_assert ((dsize % (esize * 4)) == 0);

  memset (cinfo, 0, sizeof (tensor_crop_info_s));
  
  g_assert (dsize / (esize * 4) == 1);
  g_assert (meta.type == _NNS_UINT32);

  pos = (guint *)(map.data + hsize);
  cinfo->x = pos[0];
  cinfo->y = pos[1];
  cinfo->w = pos[2];
  cinfo->h = pos[3];

  ret = TRUE;

done:
  gst_memory_unmap (mem, &map);
  return ret;
}

/**
 * @brief Internal function to crop incoming buffer.
 */
static GstBuffer *
gst_crop_scale_do_scale (GstCropScale * self, GstBuffer * raw,
    GstVideoInfo *vinfo, tensor_crop_info_s * cinfo)
{
  GstBuffer *result;
  GstMemory *mem;
  GstMapInfo map;
  guint i, j, w_inp, h_inp;
  gsize size;
  guint8 *scaled, *ptr, *inp, *row_ptr, *row_inp, *pix_inp;
  guint height, width;

  i = gst_buffer_n_memory (raw);
  g_assert (i > 0);

  mem = gst_buffer_peek_memory (raw, 0);
  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    GST_ERROR_OBJECT (self, "Failed to map the raw buffer.");
    return NULL;
  }

  height = vinfo->height;
  width = vinfo->width;

  size = 4 * width * height;
  g_assert (size == map.size);

  result = gst_buffer_new ();

  scaled = g_malloc0 (size);

  /* neareast-neighbor */
  ptr = scaled + 4 * width * cinfo->y + 4 * cinfo->x;
  inp = (guint8 *)map.data;
  for (i = 0; i < cinfo->h; i++) {
    h_inp = (int)((float)height / cinfo->h * i);
    row_inp = inp + 4 * width * h_inp;
    row_ptr = ptr;
    for (j = 0; j < cinfo->w; j++) {
      w_inp = (int)((float)width / cinfo->w * j);
      pix_inp = row_inp + 4 * w_inp;
      memcpy (row_ptr, pix_inp, 4);
      row_ptr += 4;
    }
    ptr += 4 * width;
  }

  gst_buffer_append_memory (result, 
        gst_memory_new_wrapped (0, scaled, size, 0, size, scaled, g_free));
  gst_buffer_copy_into (result, raw, GST_BUFFER_COPY_METADATA, 0, -1);

  gst_memory_unmap (mem, &map);
  return result;
}

/**
 * @brief Internal function to transform the input buffer.
 */
static GstFlowReturn
gst_crop_scale_chain (GstCropScale * self,
    GstCollectData * data_raw, GstCollectData * data_info)
{
  GstFlowReturn ret;
  GstBuffer *buf_raw, *buf_info, *result;
  GstCropScalePadData *cpad;
  GstVideoInfo *vinfo;
  tensor_crop_info_s cinfo;
  gboolean drop_raw, drop_info;

  UNUSED (cpad);

  g_return_val_if_fail (data_raw && data_info, GST_FLOW_ERROR);

  buf_raw = gst_collect_pads_peek (self->collect, data_raw);
  buf_info = gst_collect_pads_peek (self->collect, data_info);
  drop_raw = (buf_raw != NULL);
  drop_info = (buf_info != NULL);

  if (!buf_raw || !buf_info) {
    ret = GST_FLOW_EOS;
    goto done;
  }

  cpad = (GstCropScalePadData *) data_raw;
  vinfo = &cpad->info;

  //cpad = (GstCropScalePadData *) data_info;
  //buf_info = gst_tensor_buffer_from_config (buf_info, &cpad->config);

  if (!gst_crop_scale_get_crop_info (self, buf_info, &cinfo)) {
    ret = GST_FLOW_ERROR;
    goto done;
  }

  result = gst_crop_scale_do_scale (self, buf_raw, vinfo, &cinfo);
  ret = gst_pad_push (self->srcpad, result);

done:
  if (buf_raw)
    gst_buffer_unref (buf_raw);
  if (buf_info)
    gst_buffer_unref (buf_info);

  /* clear buffer in collect pads */
  if (drop_raw)
    gst_buffer_unref (gst_collect_pads_pop (self->collect, data_raw));
  if (drop_info)
    gst_buffer_unref (gst_collect_pads_pop (self->collect, data_info));

  return ret;
}

/**
 * @brief Chain function called when the buffer is available on all of the collect pads.
 */
static GstFlowReturn
gst_crop_scale_collected (GstCollectPads * pads, gpointer user_data)
{
  GstCropScale *self;
  GstCollectData *data_raw, *data_info;
  GSList *walk;
  GstFlowReturn ret;

  self = GST_CROP_SCALE (user_data);
  data_raw = data_info = NULL;

  ret = gst_crop_scale_negotiate (self);
  if (ret != GST_FLOW_OK)
    return ret;

  for (walk = pads->data; walk; walk = g_slist_next (walk)) {
    GstCollectData *data;

    data = (GstCollectData *) walk->data;

    if (data->pad == self->sinkpad_raw) {
      data_raw = data;
    } else if (data->pad == self->sinkpad_info) {
      data_info = data;
    }
  }

  return gst_crop_scale_chain (self, data_raw, data_info);
}


/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
cropscale_init (GstPlugin * cropscale)
{
  /* debug category for filtering log messages
   *
   * exchange the string 'Template cropscale' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_crop_scale_debug, "cropscale",
      0, "Template cropscale");

  return GST_ELEMENT_REGISTER (crop_scale, cropscale);
}

/* PACKAGE: this is usually set by meson depending on some _INIT macro
 * in meson.build and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use meson to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "myfirstcropscale"
#endif

/* gstreamer looks for this structure to register cropscales
 *
 * exchange the string 'Template cropscale' with your cropscale description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    cropscale,
    "crop_scale",
    cropscale_init,
    "0.0.1", "NONE", "CropScale", "NONE")
