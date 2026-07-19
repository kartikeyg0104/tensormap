/**
 * ExportPanel component for downloading trained models in different formats.
 *
 * Features:
 * - Lists available export formats (SavedModel, TFLite, ONNX)
 * - Shows file sizes and expiry dates
 * - Handles ONNX compatibility issues gracefully
 * - Downloads exports on button click
 * - Shows loading and error states
 */

import { useState, useEffect } from "react";
import { Download, Package, AlertCircle, CheckCircle, Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import axios from "../../shared/Axios";
import * as urls from "../../constants/Urls";
import logger from "../../shared/logger";
import type { ExportFormat, ExportFormatsResponse, ExportError } from "../../types/export";

interface ExportPanelProps {
  jobId: string;
  modelName: string;
}

interface FormatDisplay {
  format: ExportFormat;
  label: string;
  description: string;
  icon: typeof Package;
}

const FORMAT_INFO: FormatDisplay[] = [
  {
    format: "savedmodel",
    label: "SavedModel",
    description: "TensorFlow SavedModel format (directory structure, zipped)",
    icon: Package,
  },
  {
    format: "tflite",
    label: "TFLite",
    description: "TensorFlow Lite format for mobile/edge deployment",
    icon: Package,
  },
  {
    format: "onnx",
    label: "ONNX",
    description: "Open Neural Network Exchange format for cross-framework compatibility",
    icon: Package,
  },
];

export default function ExportPanel({ jobId, modelName }: ExportPanelProps) {
  const [formats, setFormats] = useState<ExportFormatsResponse["formats"] | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<Record<ExportFormat, boolean>>({
    savedmodel: false,
    tflite: false,
    onnx: false,
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchFormats();
  }, [jobId]);

  const fetchFormats = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get<ExportFormatsResponse>(
        `${urls.BACKEND_MODEL_EXPORT}/${jobId}/formats`,
      );
      setFormats(response.data.formats);
    } catch (err: any) {
      logger.error("Failed to fetch export formats:", err);
      setError(err.response?.data?.message || err.message || "Failed to load export formats");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (format: ExportFormat) => {
    try {
      setDownloading((prev) => ({ ...prev, [format]: true }));
      setError(null);

      const response = await axios.get(`${urls.BACKEND_MODEL_EXPORT}/${jobId}`, {
        params: { format },
        responseType: "blob",
      });

      // Extract filename from Content-Disposition header or generate one
      const contentDisposition = response.headers["content-disposition"];
      let filename: string;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        } else {
          // Fallback with correct extension
          filename = format === "savedmodel" 
            ? `${modelName}.savedmodel.zip` 
            : `${modelName}.${format}`;
        }
      } else {
        // Fallback with correct extension
        filename = format === "savedmodel" 
          ? `${modelName}.savedmodel.zip` 
          : `${modelName}.${format}`;
      }

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      // Refresh formats to update "available" status
      await fetchFormats();
    } catch (err: any) {
      logger.error(`Failed to download ${format} export:`, err);

      // Check if it's an ONNX compatibility error
      if (err.response?.data && typeof err.response.data === "object") {
        try {
          // If the response is a blob, try to parse it as JSON
          const text = await new Response(err.response.data).text();
          const errorData: ExportError = JSON.parse(text);
          if (errorData.error === "onnx_unsupported") {
            setError(
              `ONNX export not supported: ${errorData.issues.join("; ")}. ${errorData.suggestion}`,
            );
            return;
          }
        } catch {
          // Not a JSON error, continue with default error handling
        }
      }

      setError(err.response?.data?.detail || err.message || `Failed to download ${format} export`);
    } finally {
      setDownloading((prev) => ({ ...prev, [format]: false }));
    }
  };

  const formatBytes = (bytes: number | null): string => {
    if (bytes === null) return "Unknown";
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  const formatExpiry = (expiresAt: string | null): string => {
    if (!expiresAt) return "Unknown";
    const date = new Date(expiresAt);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays < 0) return "Expired";
    if (diffDays === 0) return "Expires today";
    if (diffDays === 1) return "Expires tomorrow";
    return `Expires in ${diffDays} days`;
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            Export Model
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            Loading export formats...
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!formats) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            Export Model
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {error || "Failed to load export formats. Please try again."}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Package className="h-5 w-5" />
          Export Model
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid gap-4 md:grid-cols-3">
          {FORMAT_INFO.map((info) => {
            const formatData = formats[info.format];
            const isOnnx = info.format === "onnx";
            const onnxUnsupported = isOnnx && !formatData.onnx_supported;

            return (
              <div
                key={info.format}
                className="rounded-lg border p-4 space-y-3 hover:border-primary/50 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2">
                    <info.icon className="h-5 w-5 text-muted-foreground" />
                    <h3 className="font-semibold">{info.label}</h3>
                  </div>
                  {formatData.available && <CheckCircle className="h-4 w-4 text-green-600" />}
                </div>

                <p className="text-sm text-muted-foreground">{info.description}</p>

                {onnxUnsupported && formatData.onnx_issues && (
                  <Alert variant="default" className="py-2">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-xs">
                      {formatData.onnx_issues.join(" ")}
                    </AlertDescription>
                  </Alert>
                )}

                <div className="space-y-1 text-xs text-muted-foreground">
                  {formatData.available && (
                    <>
                      <div className="flex items-center gap-1">
                        <Package className="h-3 w-3" />
                        <span>Size: {formatBytes(formatData.size_bytes)}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        <span>{formatExpiry(formatData.expires_at)}</span>
                      </div>
                    </>
                  )}
                </div>

                <Button
                  onClick={() => handleDownload(info.format)}
                  disabled={onnxUnsupported || downloading[info.format]}
                  variant={formatData.available ? "default" : "outline"}
                  className="w-full"
                  size="sm"
                >
                  {downloading[info.format] ? (
                    "Downloading..."
                  ) : (
                    <>
                      <Download className="h-4 w-4 mr-2" />
                      {formatData.available ? "Download" : "Generate & Download"}
                    </>
                  )}
                </Button>
              </div>
            );
          })}
        </div>

        <div className="text-xs text-muted-foreground border-t pt-4">
          <p>
            <strong>Note:</strong> Exports are generated on-demand and cached for 7 days. First
            download may take a few seconds.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
