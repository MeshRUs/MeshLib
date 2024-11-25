using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Text;
using static MR.DotNet.AffineXf3f;
using static MR.DotNet.Box3f;
using static MR.DotNet.Vector3f;

namespace MR.DotNet
{
    using VertBitSetReadOnly = BitSetReadOnly;
    using VertCoordsReadOnly = System.Collections.ObjectModel.ReadOnlyCollection<Vector3f>;
    using VertCoords = System.Collections.Generic.List<Vector3f>;

    public class PointCloud : MeshOrPoints, IDisposable
    {
        
        /// creates a new PointCloud object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrPointCloudNew();

        /// creates a new point cloud from existing points
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrPointCloudFromPoints( IntPtr points, ulong pointsNum );

        /// coordinates of points
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrPointCloudPoints( IntPtr pc );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrPointCloudPointsRef(IntPtr pc);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrPointCloudPointsNum( IntPtr pc );

        /// unit normal directions of points (can be empty if no normals are known)
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrPointCloudNormals( IntPtr pc );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrPointCloudNormalsNum( IntPtr pc );

        /// only points and normals corresponding to set bits here are valid
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrPointCloudValidPoints( IntPtr pc );

        /// passes through all valid points and finds the minimal bounding box containing all of them;
        /// if toWorld transformation is given then returns minimal bounding box in world space
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRBox3f mrPointCloudComputeBoundingBox( IntPtr pc, IntPtr toWorld );

        /// appends a point and returns its VertId
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern VertId mrPointCloudAddPoint(IntPtr pc, ref MRVector3f point );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern VertId mrPointCloudAddPointWithNormal(IntPtr pc, ref MRVector3f point, ref MRVector3f normal );

        /// deallocates a PointCloud object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrPointCloudFree(IntPtr pc);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern IntPtr mrPointsLoadFromAnySupportedFormat( string filename, IntPtr* errorString );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern void mrPointsSaveToAnySupportedFormat( IntPtr pc, string file, IntPtr* errorString );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrLoadIOExtras();

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        /// creates a new PointCloud object
        public PointCloud()
        {
            pc_ = mrPointCloudNew();
        }

        internal PointCloud(IntPtr pc)
        {
            pc_ = pc;
        }

        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if ( disposing )
            {
                if ( validPoints_ is not null )
                {
                    validPoints_.Dispose();
                    validPoints_ = null;
                }
            }

            if (!disposed)
            {
                if (pc_ != IntPtr.Zero)
                {
                    mrPointCloudFree(pc_);
                }

                disposed = true;
            }
        }

        ~PointCloud()
        {
            mrPointCloudFree(pc_);
        }
        /// returns point coordinates
        public VertCoordsReadOnly Points 
        { 
            get
            {
                if (points_ is null)
                {
                    int numPoints = (int)mrPointCloudPointsNum(pc_);
                    points_ = new VertCoords(numPoints);
                    int sizeOfVector3f = Marshal.SizeOf(typeof(MRVector3f));

                    var pointsPtr = mrPointCloudPoints(pc_);
                    for (int i = 0; i < numPoints; i++)
                    {
                        IntPtr currentPointPtr = IntPtr.Add(pointsPtr, i * sizeOfVector3f);
                        var point = Marshal.PtrToStructure<MRVector3f>(currentPointPtr);
                        points_.Add(new Vector3f(point));
                    }
                }

                return points_.AsReadOnly();
            } 
        }
        /// set of all valid vertices
        public VertBitSetReadOnly ValidPoints
        {
            get
            {
                if (validPoints_ is null)
                {
                    validPoints_ = new VertBitSet(mrPointCloudValidPoints(pc_));
                }
                return validPoints_;
            }
        }

        public Box3f BoundingBox
        {
            get
            {
                if (boundingBox_ is null)
                {
                    boundingBox_ = new Box3f(mrPointCloudComputeBoundingBox(pc_, (IntPtr)null));
                }

                return boundingBox_;
            }
        }
        /// returns point normals
        public VertCoordsReadOnly Normals
        {
            get
            {
                if (normals_ is null)
                {
                    int numPoints = (int)mrPointCloudNormalsNum(pc_);
                    normals_ = new VertCoords(numPoints);
                    int sizeOfVector3f = Marshal.SizeOf(typeof(MRVector3f));

                    var normalsPtr = mrPointCloudNormals(pc_);
                    for (int i = 0; i < numPoints; i++)
                    {
                        IntPtr currentPointPtr = IntPtr.Add(normalsPtr, i * sizeOfVector3f);
                        var point = Marshal.PtrToStructure<MRVector3f>(currentPointPtr);
                        normals_.Add(new Vector3f(point));
                    }
                }

                return normals_.AsReadOnly();
            }
        }
        /// loads point cloud from file of any supported format
        unsafe public static PointCloud FromAnySupportedFormat(string path)
        {
            mrLoadIOExtras();

            IntPtr errString = new IntPtr();
            var mesh = mrPointsLoadFromAnySupportedFormat(path, &errString);

            if (errString != IntPtr.Zero)
            {
                var errData = mrStringData(errString);
                string errorMessage = Marshal.PtrToStringAnsi(errData);
                throw new SystemException(errorMessage);
            }

            return new PointCloud(mesh);
        }
        /// saves point cloud to file of any supported format
        unsafe public static void ToAnySupportedFormat(PointCloud pc, string path)
        {
            mrLoadIOExtras();

            IntPtr errString = new IntPtr();
            mrPointsSaveToAnySupportedFormat(pc.pc_, path, &errString);
            if (errString != IntPtr.Zero)
            {
                var errData = mrStringData(errString);
                string errorMessage = Marshal.PtrToStringAnsi(errData);
                throw new SystemException(errorMessage);
            }
        }
        /// appends a point
        public void AddPoint(Vector3f point )
        {
            if (mrPointCloudNormalsNum(pc_) > 0)
                throw new InvalidOperationException("Normals must be empty");

            mrPointCloudAddPoint(pc_, ref point.vec_);
        }
        /// appends a point and a normal
        public void AddPoint(Vector3f point, Vector3f normal)
        {
            if (mrPointCloudNormalsNum(pc_) != mrPointCloudPointsNum(pc_))
                throw new InvalidOperationException("Points and normals must have the same size");

            mrPointCloudAddPointWithNormal(pc_, ref point.vec_, ref normal.vec_);
        }

        internal IntPtr pc_;
        private VertCoords? points_;
        private VertCoords? normals_;
        private VertBitSet? validPoints_;
        private Box3f? boundingBox_;
    }
}
