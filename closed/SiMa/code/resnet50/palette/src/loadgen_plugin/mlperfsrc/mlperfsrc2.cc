/*
 * GStreamer
 * Copyright (C) 2022 SiMa.ai
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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <chrono>

#include <glib.h>
#include <gst/gst.h>

#include <simaai/gstsimaaiallocator.h>
#include <simaai/parser_types.h>
#include <simaai/parser.h>
#include <simaai/simaai_memory.h>

#include "mlperfsrc2.h"
#include "loadgen_plugin.h"

/**
 * @brief dim[3]:dim[2]:dim[1]:dim[0]
 * N:C:H:W
 */
typedef struct _TensorDim {
  gint32 dim[6]; ///< The NCHW representation of the tensor dimension
} TensorDim;

struct _GstMlperfSrcPrivate
{
  GstAllocator * allocator; ///< gst allocator handle 
  GstMemory * mem; ///< The opaque pointer to simaai memory
  GThread * worker_thread; ///< worker thread to get data from mlperf imagenet
  GString *config_file;

  gint64 buf_id; ///< simaai memory buffer id, returned after allocation. Passed onto next plugin
  gchar *filename; ///< filename of where the input data is available
  guint64 scenario; ///< Scenario type, {0,1,2} {SingleStream, MultiStream, Offline}
  gboolean toy_mode; ///< Toy mode, runs with 10000 images
  gsize bytes_available; ///< The number of bytes read from imagenet. Should match tensor_dims
  gint64 frame_id; ///< The sequence id incremented for every iteration
  
  TensorDim tensor_dim; ///< Refer above the tensor dimesnion in NCHW format
  guint64 run_type; ////< 0=Performance or 1=Accuracy

  simaai_params_t * params;
};

#define UNUSED(x) (void)(x)
#define MLPERF_DEFAULT_SIZE (1)
#define MLPERFSRC_MAX_TOKENS 4
#define CONFIG_FILE "config.json"

GST_DEBUG_CATEGORY_STATIC (gst_mlperfsrc_debug);
#define GST_CAT_DEFAULT gst_mlperfsrc_debug

/**
 * @brief the capabilities of the outputs 
 */
static GstStaticPadTemplate srctemplate = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

/**
 * @brief mlperfsrc properties
 */
enum
{
  PROP_0,
  PROP_CONF_F,
  PROP_LOCATION,
  PROP_TOY_MODE,
  PROP_SCENARIO,
  PROP_TENSOR_DIMS,
  PROP_RUN_TYPE,
  PROP_LAST
};

#define gst_mlperfsrc_parent_class parent_class
G_DEFINE_TYPE (GstMlperfSrc, gst_mlperfsrc, GST_TYPE_BASE_SRC);

static void gst_mlperfsrc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_mlperfsrc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_mlperfsrc_class_finalize (GObject * object);

static gboolean gst_mlperfsrc_start (GstBaseSrc * basesrc);
static GstFlowReturn gst_mlperfsrc_create (GstBaseSrc * basesrc, guint64 offset,
    guint size, GstBuffer ** out_buf);

static int32_t
gst_mlperf_dump_output_buffer (GstMlperfSrc * self, gsize offset);

static gint
gst_mlperfsrc_validate_data_size (GstMlperfSrc * self, const gsize sz);

static gboolean
gst_mlperfsrc_set_tensor_dimension (GstMlperfSrc * self, const gchar * dimstr);

static gsize
gst_mlperfsrc_get_tensor_size (GstMlperfSrc * self);

static void gst_simaai_buffer_recvd_cb(gpointer buffer, void * priv)
{
  return;
}

/**
 * @brief initialize the class
 */
static void
gst_mlperfsrc_class_init (GstMlperfSrcClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
  GstBaseSrcClass *gstbasesrc_class = GST_BASE_SRC_CLASS (klass);

  gobject_class->set_property = gst_mlperfsrc_set_property;
  gobject_class->get_property = gst_mlperfsrc_get_property;
  gobject_class->finalize = gst_mlperfsrc_class_finalize;

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&srctemplate));

  g_object_class_install_property (gobject_class, PROP_CONF_F,
                                   g_param_spec_string ("config",
                                                        "ConfigFile",
                                                        "Config JSON to be used",
                                                        CONFIG_FILE,
                                                        (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_LOCATION,
                                   g_param_spec_string ("inpath", "File Path",
                                                        "Input dat to read", NULL,
                                                        GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_SCENARIO,
                                   g_param_spec_uint ("mlperf-scenario", "Scenario",
                                                      "0 = SingleStream, 1 = MultiStream, 2 = Offline",
                                                      0, 3, 0,
                                                      GParamFlags(G_PARAM_READWRITE |
                                                                  GST_PARAM_MUTABLE_PLAYING |
                                                                  G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_TOY_MODE,
                                   g_param_spec_boolean ("toy-mode", "Enable Toy Mode",
                                                         "Toy Mode uses fewer images to run faster and easier",
                                                         false, GParamFlags(G_PARAM_READWRITE |
                                                                            GST_PARAM_MUTABLE_PLAYING |
                                                                            G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_RUN_TYPE,
                                   g_param_spec_uint ("mlperf-run-type", "Run Type",
                                                      "0=Perf, 1=Acc, 2=Power",
                                                      0, 3, 0,
                                                      GParamFlags(G_PARAM_READWRITE |
                                                                  GST_PARAM_MUTABLE_PLAYING |
                                                                  G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_TENSOR_DIMS,
                                   g_param_spec_string ("dims", "Tensor Dimension as N:C:H:W",
                                                        "Tensor Dimension as BatchSize:ChannelWidth:Height:Width", NULL,
                                                        GParamFlags(G_PARAM_READWRITE |
                                                                    G_PARAM_STATIC_STRINGS |
                                                                    GST_PARAM_MUTABLE_READY)));

  static const gchar *tags[] = { NULL };
  gst_meta_register_custom ("GstSimaMeta", tags, NULL, NULL, NULL);

  gst_element_class_set_static_metadata (gstelement_class,
      "MlperfSrc", "Source MLper",
      "Read image and push", "SiMa.Ai");

  gstbasesrc_class->start = gst_mlperfsrc_start;
  gstbasesrc_class->create = gst_mlperfsrc_create;
  
  GST_DEBUG_CATEGORY_INIT (GST_CAT_DEFAULT,
      "mlperfsrc", 0, "Mlperf src");
}

/**
 * @brief initialize mlperfsrc element 
 */
static void
gst_mlperfsrc_init (GstMlperfSrc * self)
{
  GstBaseSrc *basesrc = GST_BASE_SRC (self);

  gst_simaai_buffer_memory_init_once();
  
  gst_base_src_set_format (basesrc, GST_FORMAT_TIME);
  gst_base_src_set_async (basesrc, FALSE);

  self->data_queue = g_async_queue_new ();

  self->priv = (GstMlperfSrcPrivate *) g_malloc(sizeof(GstMlperfSrcPrivate));

  self->priv->filename = NULL;
  self->priv->frame_id = 0;

  // self->priv->buf_cb.buffer_received = gst_simaai_buffer_recvd_cb;
  // self->priv->buf_cb.priv = self;
  
  // self->priv->p_buf_exchanger =
  //     buffer_data_exchanger_new(&self->priv->buf_cb);

  self->priv->config_file = g_string_new(CONFIG_FILE);
}

/**
 * @brief set property
 */
static void
gst_mlperfsrc_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstMlperfSrc *self = GST_MLPERFSRC (object);

  switch (prop_id) {
    case PROP_CONF_F:
        g_string_assign(self->priv->config_file, g_value_get_string(value));
        GST_DEBUG_OBJECT(self, "Config argument was changed to %s", self->priv->config_file->str);
        break;
    case PROP_LOCATION:
      self->priv->filename = g_strdup (g_value_get_string(value));
      break;
    case PROP_TOY_MODE:
      self->priv->toy_mode = g_value_get_boolean (value);
      break;
    case PROP_SCENARIO:
      self->priv->scenario = g_value_get_uint (value);
      break;
    case PROP_RUN_TYPE:
      self->priv->run_type = g_value_get_uint (value);
      break;
    case PROP_TENSOR_DIMS:
      gst_mlperfsrc_set_tensor_dimension (self, g_value_get_string(value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief get property
 */
static void
gst_mlperfsrc_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstMlperfSrc *self = GST_MLPERFSRC (object);

  switch (prop_id) {
    case PROP_LOCATION:
      g_value_set_string (value, self->priv->filename);
      break;
    case PROP_CONF_F:
        g_value_set_string(value, (const gchar *)self->priv->config_file);
        break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief finalize the object
 */
static void
gst_mlperfsrc_class_finalize (GObject * object)
{
  GstMlperfSrc *self = GST_MLPERFSRC (object);
  int8_t * data_h;

  g_string_free(self->priv->config_file, TRUE);
  g_free (self->priv->filename);

  simaai_memory_free(simaai_get_memory_handle(self->priv->mem));

  if (self->data_queue) {
    while ((data_h = (int8_t *) g_async_queue_try_pop (self->data_queue))) {
      // TODO: Free data from mlperf?
    }
    g_async_queue_unref (self->data_queue);
    self->data_queue = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief mlperfsrc data callback.
 */
static gpointer
_mlperf_data_cb (gpointer user_data)
{
  int ret = 0;
  GstMlperfSrc * self = GST_MLPERFSRC (user_data);

  while (true) {
    gsize bytes_ready = 0;

    int8_t * data = (int8_t *) mlperf_block_for_next_example(&bytes_ready);
    self->priv->bytes_available = bytes_ready;

    if ((data == NULL) || (bytes_ready <= 0)) {
      g_async_queue_push (self->data_queue, data);
      break;
    }

    if (!gst_mlperfsrc_validate_data_size(self, bytes_ready)) {
      g_error("Tensor dimension does'nt match the data size read");
      g_assert(bytes_ready);
      break;
    }
    g_async_queue_push (self->data_queue, data);
  }

  return NULL;
}

/**
 * @brief start mlperfsrc, called when state changed null to ready
 */
static gboolean
gst_mlperfsrc_start (GstBaseSrc * basesrc)
{
  GstMlperfSrc *self = GST_MLPERFSRC (basesrc);

  gsize tensor_size = gst_mlperfsrc_get_tensor_size(self);

  self->priv->buf_id = 0;
  self->priv->mem = simaai_target_mem_alloc(SIMAAI_MEM_TARGET_MOSAIC,
                                           tensor_size,
                                           &self->priv->buf_id);

  GST_DEBUG_OBJECT(self, "Initialized simaaimemory , with buffer id : %ld, size: %ld", self->priv->buf_id, tensor_size);

  if (self->priv->mem == NULL) {
    g_error("ERROR: allocating contiguous memory for out_buf of size ");
    g_assert(self->priv->mem == NULL);
  }

  mlperf_initialize(self->priv->filename, self->priv->scenario, self->priv->run_type, self->priv->toy_mode);
  mlperf_start();

  return TRUE;
}

/**
 * @brief Create a buffer containing the subscribed data
 */
static GstFlowReturn
gst_mlperfsrc_create (GstBaseSrc * basesrc, guint64 offset, guint size,
    GstBuffer ** out_buf)
{
  GstMlperfSrc *self = GST_MLPERFSRC (basesrc);

  GstBuffer *buffer = NULL;
  int ret;

  UNUSED (offset);
  UNUSED (size);

  gsize bytes_ready = 0;
  int8_t * data_h = (int8_t *) mlperf_block_for_next_example(&bytes_ready); // mlperf_block_for_next_example(&self->priv->bytes_available);
  // 
  self->priv->bytes_available = bytes_ready;
  if ((self->priv->bytes_available <= 0) || (data_h == NULL)) {
    g_message("Sending EOS");
    return GST_FLOW_EOS;
  }

  gsize data_len = self->priv->bytes_available;
  
  if (!gst_mlperfsrc_validate_data_size(self, data_len)) {
    g_error("Tensor dimension does'nt match the data size read");
    g_assert(self->priv->bytes_available);
  }
  
  GstMapInfo map;
  gst_memory_map(self->priv->mem, &map, GST_MAP_WRITE);

  if ((data_h == NULL) || (data_len <= 0)) {
      g_message("data_h is NULL %p", data_h);
      return GST_FLOW_ERROR;
  }

  memcpy((int8_t *)map.data, data_h, data_len);
    
  gst_memory_unmap(self->priv->mem, &map);

  buffer = gst_buffer_new ();

  GstCustomMeta * meta = gst_buffer_add_custom_meta(buffer, "GstSimaMeta");
  if (meta == NULL) {
    GST_ERROR ("MLPERFSRC:Unable to add metadata info to the buffer");
    return GST_FLOW_ERROR;
  }

  char node_name[] = "mlperfsrc";

  GstStructure *s = gst_custom_meta_get_structure (meta);
  if (s != NULL) {
    gst_structure_set (s,
                       "buffer-id", G_TYPE_INT64, self->priv->buf_id,
                       "buffer-name", G_TYPE_STRING, node_name,
                       "buffer-offset", G_TYPE_INT64, 0,
                       "frame-id", G_TYPE_INT64, self->priv->frame_id++, NULL);
  } else {
       g_message("MLPERFSRC: Unable to add metadata to the buffer");
       return GST_FLOW_ERROR;
  }
  
  gst_buffer_append_memory (buffer,
                            gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
                                                    map.data,
                                                    map.size,
                                                    0,
                                                    data_len,
                                                    map.data,
                                                    g_free));
done:
  if (buffer == NULL) {
    GST_ERROR ("Failed to get buffer to push to the mlperfsrc.");
    return GST_FLOW_ERROR;
  }

  *out_buf = buffer;
  return GST_FLOW_OK;
}

static int32_t
gst_mlperf_dump_output_buffer (GstMlperfSrc * self, gsize offset)
{
  FILE *ofp;
  size_t wosz;
  void *ovaddr;
  int32_t res = 0, i;
  char *dumpaddr;
  char full_opath[256];

  snprintf(full_opath, sizeof(full_opath) - 1, "/tmp/%s-%ld.out", "mlperfsrc", self->priv->frame_id);

  ofp = fopen(full_opath, "w");
  if(ofp == NULL) {
    res = -1;
    goto err_ofp;
  }

  GstMapInfo map;
  gst_memory_map(self->priv->mem, &map, GST_MAP_READ);
  wosz = fwrite((char *)(map.data), 1, 150528, ofp);
  if (wosz != 150528) {
    res = -3;
    goto err_owrite;
  }
  gst_memory_unmap(self->priv->mem, &map);     // Reinitialize these somewhere else

err_owrite:
err_omap:
  fclose(ofp);
err_ofp:
  return res;
}

static gint
gst_mlperfsrc_validate_data_size (GstMlperfSrc * self, const gsize sz) {
  return ((((gint) sz % (gint) gst_mlperfsrc_get_tensor_size(self)) == 0) ? 1: 0);
}

static gboolean
gst_mlperfsrc_set_tensor_dimension (GstMlperfSrc * self, const gchar * dimstr)
{
  guint rank = 0;
  guint64 val;
  gchar **strv;
  gchar *dim_string;
  guint i, num_dims;

  if (dimstr == NULL)
    return 0;

  /* remove spaces */
  dim_string = g_strdup (dimstr);
  g_strstrip (dim_string);

  strv = g_strsplit (dim_string, ":", MLPERFSRC_MAX_TOKENS);
  num_dims = g_strv_length (strv);
  if (num_dims > 4) {
    GST_ERROR("Unsupported tensor dimension by SiMa MLA");
    return FALSE;
  }
  
  for (i = 0; i < num_dims; i++) {
    g_strstrip (strv[i]);
    if (strv[i] == NULL || strlen (strv[i]) == 0)
      break;

    val = g_ascii_strtoull (strv[i], NULL, 10);
    self->priv->tensor_dim.dim[i] = (uint32_t) val;
    rank = i + 1;
  }

  for (; i < MLPERFSRC_MAX_TOKENS; i++)
    self->priv->tensor_dim.dim[i] = 1;

  g_strfreev (strv);
  g_free (dim_string);
  return rank;
}

static gsize
gst_mlperfsrc_get_tensor_size (GstMlperfSrc * self)
{
  gsize size = MLPERF_DEFAULT_SIZE;

  for (int i = 0; i < 4 ; i++)
    size *= self->priv->tensor_dim.dim[i];

  return size;
}

static gboolean
plugin_init (GstPlugin *plugin)
{
  g_message ("MLPERFSRC: Calling pugin Init Function...");
  if (!gst_element_register(plugin, "mlperfsrc2", GST_RANK_NONE,
                            GST_TYPE_MLPERFSRC)) {
    GST_ERROR("Unable to register process2 plugin");
    return FALSE;
  }

  return TRUE;
}

GST_PLUGIN_DEFINE (
  GST_VERSION_MAJOR,
  GST_VERSION_MINOR,
  simaaimlperfsrc,
  "SiMa MLPerf Src",
  plugin_init,
  "0.1",
  "unknown",
  "GStreamer",
  "SiMa.ai"
)
